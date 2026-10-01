"""Frozen M8 scientific model methods with narrow operational bindings.

No class construction at import. NativeBackend is only constructed by explicit
future execution after all local assets and new-run identity pass validation.
"""
from __future__ import annotations
import contextlib,os,time,json
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np
import torch
from petitzero.countdown import check_expression
from petitzero.grpo import selected_action_logps
from petitzero.ppo import prefix_values
from petitzero.ppo_runtime import compact_batch,terminal_kind
from petitzero.rft_runtime import run_optimizer_slot,select_responses
from petitzero.v2.methods import objective,prepare_targets
from petitzero.v2.profile import fm_profile,assert_fm,metadata
from petitzero.v2.study_utils import tensor_hash,fingerprint,seed_all,rng_state,restore_rng,describe,alignment,finalize_metrics,verify_optimizer,verify_update_change
from .protocol import digest,require,CHECKPOINT
from .durable import atomic,append

class AlignmentFailure(RuntimeError):pass

class DurableAdamW(torch.optim.AdamW):
    def __init__(self,parameters,owner,role,lr):
        owner.journal.add('optimizer_construction_start',role=role)
        super().__init__(parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.owner=owner;self.role=role
    def step(self,closure=None):
        o=self.owner
        assert self.role in o.steps and o.steps[self.role]<o.context['spec']['caps']['actor_slots']
        o.journal.add('optimizer_step_start',role=self.role,slot=o.current_slot)
        with o.measure(self.role+'_optimizer_step'),fm_profile():
            value=super().step(closure);o.steps[self.role]+=1
            verify_optimizer(self,o.steps[self.role])
            assert all(torch.isfinite(p).all() for group in self.param_groups for p in group['params'])
        o.journal.add('optimizer_step_commit',role=self.role,slot=o.current_slot,actual_steps=dict(o.steps))
        return value

class FrozenMethods:
    def load_model(self, role: str, adapter: Path, trainable: bool):
        self.journal.add('model_construction_start',role=role)
        self.counter['model_constructions'] += 1
        base = AutoModelForCausalLM.from_pretrained(self.args.base, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True, trust_remote_code=False, use_safetensors=True)
        assert all(p.dtype == torch.bfloat16 for p in base.parameters())
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable, autocast_adapter_dtype=True, local_files_only=True)
        if not trainable:
            model.requires_grad_(False)
        assert all(p.dtype == (torch.float32 if '.lora_' in name else torch.bfloat16) for name, p in model.named_parameters())
        model.float().to('cuda')
        model.eval()
        model.config.use_cache = False
        assert all(p.dtype == torch.float32 and p.requires_grad == (trainable and '.lora_' in name)
                   for name, p in model.named_parameters())
        assert model.active_adapters == ['default'] and not any(m.merged for m in model.modules() if hasattr(m, 'merged'))
        assert model.config.attention_dropout == 0
        assert all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
        raw = model.get_base_model()

        def wrapped_forward(module, inputs):
            assert_fm()
            if 'generation' not in self.phase:
                self.journal.add('model_forward_start',role=role,phase=self.phase)
                self.counter[f'forward/{role}/{self.phase}'] += 1

        def raw_forward(module, inputs):
            assert_fm()
            if 'generation' in self.phase:
                self.journal.add('model_forward_start',role=role,phase=self.phase)
                self.counter[f'forward/{role}/{self.phase}'] += 1

        def projection(module, inputs, output):
            assert inputs[0].dtype == output.dtype == module.weight.dtype == torch.float32
            key = (role, self.phase)
            if key not in self.dtype_seen:
                self.dtype_seen.add(key)
                self.append('dtype_evidence.jsonl', {'role': role, 'phase': self.phase, 'attempt': self.args.attempt,
                            'projection_input': str(inputs[0].dtype), 'logits': str(output.dtype), 'profile': metadata()})

        def checkpoint_layer(module, inputs):
            if self.backward_role:
                assert_fm()
                self.journal.add('checkpoint_layer_recomputation',role=role)
                self.counter[f'checkpoint_layer_recomputations/{role}'] += 1

        self.handles += [model.register_forward_pre_hook(wrapped_forward), raw.register_forward_pre_hook(raw_forward), raw.lm_head.register_forward_hook(projection)]
        self.handles += [layer.register_forward_pre_hook(checkpoint_layer) for layer in raw.model.layers]
        original_processors = raw._get_logits_processor

        def checked_processors(*args, **kwargs):
            processors = original_processors(*args, **kwargs)
            assert not processors, 'Distribution-changing logits processor'
            self.counter['empty_processor_lists'] += 1
            return processors

        raw._get_logits_processor = checked_processors
        return model

    def adapter_hashes(self, role: str) -> dict:
        return {name: tensor_hash(p) for name, p in self.models[role].named_parameters() if '.lora_' in name}

    def assert_independent(self) -> None:
        pointers = {role: {p.untyped_storage().data_ptr() for p in model.parameters()} for role, model in self.models.items()}
        for left in pointers:
            for right in pointers:
                if left != right:
                    assert pointers[left].isdisjoint(pointers[right])
        if 'reference' in self.models:
            assert all(not p.requires_grad for p in self.models['reference'].parameters())

    def check_frozen(self) -> None:
        for role, model in self.models.items():
            assert {name: p._version for name, p in model.named_parameters() if not p.requires_grad} == self.frozen_versions[role]
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)

    def score(self, role: str, records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        batch, positions, mask = compact_batch(records, 'cuda')
        labels = batch.pop('labels')
        assert batch['input_ids'].dtype == batch['attention_mask'].dtype == positions.dtype == torch.long
        self.last_shape = {'role': role, 'input_shape': list(batch['input_ids'].shape),
                           'action_lengths': [len(r['output_ids']) for r in records]}
        model = self.models[role]
        model.config.use_cache = False
        with fm_profile():
            output = model(**batch, use_cache=False, output_hidden_states=role == 'critic')
            if role == 'critic':
                values = prefix_values(output.hidden_states[-1], positions, mask, self.head)
            else:
                full, full_mask = selected_action_logps(output.logits, labels)
                assert full_mask.sum() == mask.sum()
                values = torch.where(mask, full.gather(1, positions), 0.)
        assert values.dtype == torch.float32 and bool(torch.isfinite(values[mask]).all())
        return values, mask

    def backward(self, role: str, records: list[dict], totals: dict, pass_index: int) -> dict:
        numeric = Counter()
        micro_losses, ratios, current_rows = [], [], []
        with self.measure(role + '_backward'), fm_profile():
            for offset in range(0, len(records), 2):
                microbatch = records[offset:offset + 2]
                current, mask = self.score(role, microbatch)
                loss, metrics = objective(self.method, role, current, mask, microbatch, totals)
                assert loss.requires_grad and bool(torch.isfinite(loss))
                numeric['loss'] += loss.detach().item()
                micro_losses.append(loss.detach().item())
                for key, value in metrics.items():
                    if value.numel() == 1:
                        numeric[key] += value.detach().item()
                if role == 'actor' and self.method in ['grpo', 'ppo', 'ppo_zero']:
                    from petitzero.ppo_runtime import stored
                    old = stored(microbatch, 'old_action_logps', mask)
                    ratios.extend((current.detach() - old).exp()[mask].cpu().tolist())
                    if pass_index == 1:
                        for index, record in enumerate(microbatch):
                            current_rows.append({**record, 'current_before_pass1': current[index, mask[index]].detach().cpu().tolist()})
                    del old
                self.backward_role = role
                loss.backward()
                self.backward_role = None
                self.counter[role + '_backward_calls'] += 1
                del current, mask, loss, metrics
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.parameters[role])
            if role == 'actor' and self.method in ['grpo', 'ppo', 'ppo_zero'] and pass_index == 1:
                gate = alignment(current_rows, 'current_before_pass1', 'old_action_logps')
                self.save(f'rollouts/batch{self.batch_in_progress:03d}/same_layout_before_pass1.json', gate)
                if not gate['passed']:
                    raise AlignmentFailure('Current/old pre-pass-1 disagreement; no optimizer step')
            if role == 'actor' and 'critic' in self.parameters:
                assert all(p.grad is None for p in self.parameters['critic'])
            if role == 'critic' and self.steps['critic'] == 0:
                zero = all(bool((p.grad == 0).all()) for p in self.parameters['critic'][:-2])
                assert zero
                self.save('checks/ZERO_HEAD_FIRST_BACKWARD.json', {'critic_adapter_gradient_zero': zero,
                          'head_gradient_norm': float(torch.sqrt(sum(p.grad.square().sum() for p in self.head.parameters())))})
            self.check_frozen()
        return finalize_metrics(numeric, microbatch_losses=micro_losses,
                                ratio=describe(ratios) if ratios else None)

    def optimize(self, records: list[dict], totals: dict, pass_index: int) -> None:
        """One nominal slot; RFT's helper owns its genuine empty-positive skip."""
        batch_digest = digest(records)
        self.current_slot = 2 * (self.batch_in_progress - 1) + pass_index
        metrics = {}
        rewards, positive_indices = select_responses(records)
        skipped = self.method == 'rft' and not positive_indices
        if skipped:
            result = run_optimizer_slot(self.optimizers['actor'], self.parameters['actor'], rewards,
                                        self.current_slot, lambda _: (_ for _ in ()).throw(AssertionError('skip invoked backward')))
            assert result['skipped']
            metrics['actor'] = result
            self.counter['rft_skipped_slots'] += 1
        else:
            before = {role: [p.detach().clone() for p in params] for role, params in self.parameters.items()}
            for role, model in self.models.items():
                if role == 'reference':
                    continue
                model.train()
                model.config.use_cache = False
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                assert model.is_gradient_checkpointing
            if self.method == 'rft':
                def accumulate(accepted: int) -> dict:
                    assert accepted == totals['N_positive']
                    return self.backward('actor', records, totals, pass_index)
                metrics['actor'] = run_optimizer_slot(self.optimizers['actor'], self.parameters['actor'],
                                                      rewards, self.current_slot, accumulate)
            else:
                for role, optimizer in self.optimizers.items():
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = (1e-4 if role == 'critic' else 1e-5) * min(self.current_slot / 8, 1)
                for role in self.optimizers:
                    metrics[role] = self.backward(role, records, totals, pass_index)
                    metrics[role]['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(self.parameters[role], 1., error_if_nonfinite=True))
                for role, optimizer in self.optimizers.items():
                    optimizer.step()
            for role, parameters in self.parameters.items():
                metrics[role]['update_l2'] = float(torch.sqrt(sum((p.detach() - old).double().square().sum() for p, old in zip(parameters, before[role]))))
                metrics[role]['learning_rate'] = self.optimizers[role].param_groups[0]['lr']
                if role == 'actor' and self.method in ['grpo', 'ppo', 'ppo_zero']:
                    metrics[role]['active_clip_fraction'] = metrics[role]['active_clip_count'] / totals['D']
                self.optimizers[role].zero_grad(set_to_none=True)
            del before
        self.nominal_slot = self.current_slot
        assert digest(records) == batch_digest
        self.check_frozen()
        if self.nominal_slot == 1:
            changes = {role: self.adapter_hashes(role) != self.initial_adapters[role] for role in self.models}
            if 'reference' in changes:
                assert not changes['reference']
            if 'critic' in changes:
                assert not changes['critic']
            if not skipped:
                verify_update_change(changes['actor'], metrics['actor']['update_l2'])
            self.save('checks/FIRST_UPDATE.json', {'adapter_changed': changes, 'rft_skipped': skipped,
                      'finite_optimizer_states': True, 'reference_and_base_unchanged': True,
                      'head_nonzero': any(bool((p != 0).any()) for p in self.head.parameters()) if self.head else None})
        self.save(f'rollouts/batch{self.batch_in_progress:03d}/update{pass_index}.json', {
            'nominal_slot': self.nominal_slot, 'actual_steps': dict(self.steps), 'pass': pass_index,
            'batch': self.batch_in_progress, 'batch_digest': batch_digest, 'totals': totals,
            'skipped': skipped, 'metrics': metrics, 'attempt': self.args.attempt})
        self.save_counters()
        print(f'{self.mode} {self.method} batch {self.batch_in_progress}/{len(self.schedule)} slot {self.nominal_slot} steps {self.steps} skipped={skipped}', flush=True)

    def _generate(self, row: dict, mode: str, sample_index: int, seed: int) -> dict:
        """One native cached answer, with original scores and categorical entropy."""
        assert len(self.journal.records)<self.context['spec']['caps']['responses']
        model = self.models['actor']
        model.gradient_checkpointing_disable()
        model.eval()
        model.config.use_cache = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prompt = self.prefixes[row['id']]
        self.last_shape = {'prompt_length': len(prompt), 'generation_cap': 128, 'id': row['id'],
                           'mode': mode, 'sample_index': sample_index, 'seed': seed}
        # runner has already fsynced request_start; no private historical caps.
        inputs = torch.tensor([prompt], dtype=torch.long, device='cuda')
        settings = dict(max_new_tokens=128, num_beams=1, num_return_sequences=1,
                        eos_token_id=[151645, 151643], pad_token_id=151643, use_cache=True,
                        repetition_penalty=1., do_sample=mode == 'sample')
        if mode == 'sample':
            settings.update(temperature=1., top_p=1., top_k=0)
        started = time.monotonic()
        with torch.inference_mode(), fm_profile():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                         generation_config=GenerationConfig(**settings),
                         return_dict_in_generate=True, output_scores=True)
            tokens = output.sequences[0, len(prompt):].tolist()
            logps, entropies = [], []
            for scores, token in zip(output.scores, tokens):
                assert scores.dtype == torch.float32
                log_probability = scores[0].log_softmax(-1)
                probability = log_probability.exp()
                logps.append(log_probability[token].item())
                entropies.append(float(-torch.where(probability > 0, probability * log_probability, 0.).sum()))
            assert len(tokens) == len(logps) == len(entropies)
            assert all(np.isfinite(v) for v in logps + entropies)
            if output.past_key_values is not None:
                assert all(layer.keys.dtype == layer.values.dtype == torch.float32 for layer in output.past_key_values.layers)
        terminal = terminal_kind(tokens)
        scoring_tokens = tokens[:-1] if tokens[-1] in [151645, 151643] else tokens
        text = self.tokenizer.decode(scoring_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        score = check_expression(text, row['numbers'], row['target']).as_dict()
        self.counter['generation_calls'] += 1
        self.counter['generated_tokens'] += len(tokens)
        kind = 'training' if self.phase == 'training_generation' else 'evaluation'
        self.counter[kind + '_completions'] += 1
        self.counter[self.mode + '_completions'] += 1
        self.save_counters()
        return {'id': row['id'], 'mode': mode, 'sample_index': sample_index, 'seed': seed,
                'method': self.method, 'run_mode': self.mode, 'training_seed': self.resolved_seed,
                'band': row['band'], 'sft_exposed': row.get('sft_exposed'), 'number_group': row['number_group'],
                'prompt_ids': prompt, 'output_ids': tokens, 'generated_tokens': len(tokens),
                'raw_text': self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                'scoring_text': text, 'score': score, 'terminal_kind': terminal,
                'stop_reason': 'eos' if terminal == 'NATIVE_EOS_TERMINAL' else 'length_cap',
                'generation_action_logps': logps, 'conditional_categorical_entropies': entropies,
                'conditional_entropy_mean': sum(entropies) / len(entropies),
                'generation_seconds': time.monotonic() - started, 'nominal_slot': self.nominal_slot,
                'actual_steps': dict(self.steps), 'attempt': self.args.attempt}

class NativeBackend(FrozenMethods):
    """Actual local Transformers/PEFT execution boundary, NOT run in M12A."""
    def __init__(self,context,run,journal,resume=None,identity=None):
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false')
        import importlib.metadata
        for package,version in {'torch':'2.8.0+cu126','transformers':'4.56.2','peft':'0.17.1','numpy':'2.2.6','safetensors':'0.8.0'}.items():
            require(importlib.metadata.version(package)==version,'Recorded dependency version required: '+package+'=='+version)
        global AutoModelForCausalLM,AutoTokenizer,GenerationConfig,PeftModel
        from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
        from peft import PeftModel
        self.context=context;self.run=run;self.journal=journal;self.root=context['output'];self.attempt_dir=self.root
        self.args=SimpleNamespace(base=Path(context['paths']['base']),attempt=sum(e['event']=='attempt_start' for e in journal.events))
        self.counter=Counter();self.models={};self.parameters={};self.optimizers={};self.handles=[];self.dtype_seen=set()
        self.phase='construction';self.backward_role=None;self.last_shape=None;self.method=context['spec']['method'];self.mode=context['spec']['operation'];self.resolved_seed=context['spec']['seed']
        self.nominal_slot=identity['nominal_slot'] if identity else 0;self.current_slot=self.nominal_slot;self.completed_batches=self.nominal_slot//2;self.batch_in_progress=self.completed_batches
        self.schedule=list(range(context['spec']['rollouts']));self.head=None
        training=self.mode!='evaluate';wanted=context['spec']['roles'];train_roles=[r for r in wanted if r!='reference'] if training else []
        self.steps=dict(identity['actual_steps']) if identity else {r:0 for r in train_roles}
        torch.set_num_threads(4);seed_all(self.resolved_seed)
        journal.add('tokenizer_construction_start')
        self.tokenizer=AutoTokenizer.from_pretrained(self.args.base,local_files_only=True,trust_remote_code=False)
        assert self.tokenizer.pad_token_id==151643 and self.tokenizer.convert_tokens_to_ids('<|im_end|>')==151645
        self.prefixes={r['id']:self.tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True) for r in context['rows']}
        assert all(p and len(p)+128<=1024 for p in self.prefixes.values())
        for role in wanted:
            adapter=Path(context['paths']['actor'] or context['paths']['sft']) if not training else (resume/role if resume and role!='reference' else Path(context['paths']['sft']))
            self.models[role]=self.load_model(role,adapter,role in train_roles)
            self.verify_loaded_adapter(role,adapter)
        if 'critic' in wanted:
            journal.add('value_head_construction_start')
            self.head=torch.nn.Linear(1536,1,bias=True,dtype=torch.float32,device='cuda')
            torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)
            if resume:self.head.load_state_dict(torch.load(resume/'head.pt',map_location='cuda',weights_only=True))
        for role in train_roles:
            params=[p for p in self.models[role].parameters() if p.requires_grad]
            if role=='critic':params+=list(self.head.parameters())
            self.parameters[role]=params;self.optimizers[role]=DurableAdamW(params,self,role,1e-4 if role=='critic' else 1e-5)
        assert set(self.models)==set(wanted) and set(self.optimizers)==set(train_roles)
        assert (self.head is not None)==('critic' in wanted)
        self.assert_independent()
        self.frozen_versions={role:{n:p._version for n,p in model.named_parameters() if not p.requires_grad} for role,model in self.models.items()}
        self.initial_adapters={role:self.adapter_hashes(role) for role in self.models}
        self.save(f'load_identities/attempt{self.args.attempt:04d}.json',{'roles':wanted,'adapters':self.initial_adapters,'all_FP32_unmerged':True,'head_present':self.head is not None,'optimizer_roles':list(self.optimizers)})
        if resume:self.restore(resume,identity)

    def save(self,name,value):atomic(self.root/name,value)
    def append(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);append(path,value)
    def save_counters(self):atomic(self.root/'backend_counters'/f'attempt{self.args.attempt:04d}.json',dict(self.counter))
    @contextlib.contextmanager
    def measure(self,phase):
        previous=self.phase;self.phase=phase;torch.cuda.synchronize();start=time.monotonic()
        try:yield
        finally:
            torch.cuda.synchronize();self.append('resources.jsonl',{'phase':phase,'seconds':time.monotonic()-start,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'attempt':self.args.attempt});self.phase=previous

    def generate(self,row,request):
        phase='evaluation_generation' if self.mode=='evaluate' else 'training_generation'
        with self.measure(phase):record=self._generate(row,request['mode'],request['sample_index'],request['seed'])
        c,n=len(record['prompt_ids']),len(record['output_ids'])
        record.update(request=request,completion_id=f"new/{request['index']}",rollout_batch=request['rollout'],terminated=True,bootstrap=0.,action_logit_positions=list(range(c-1,c+n-1)),action_input_positions=list(range(c,c+n)),critic_state_positions=list(range(c-1,c+n-1)))
        self.check_frozen()
        return record

    def prepare(self,records,batch):
        self.batch_in_progress=batch;assert batch==self.nominal_slot//2+1 and len(records)==32
        for model in self.models.values():model.gradient_checkpointing_disable();model.eval()
        keys={'actor':'old_action_logps','reference':'reference_action_logps','critic':'old_values'}
        for role in self.models:
            with self.measure(role+'_old_scoring'),torch.no_grad(),fm_profile():
                for offset in range(0,len(records),2):
                    current,mask=self.score(role,records[offset:offset+2])
                    for i,record in enumerate(records[offset:offset+2]):record[keys[role]]=current[i,mask[i]].cpu().tolist()
        gate=alignment(records,'old_action_logps','generation_action_logps');self.save(f'rollouts/batch{batch:03d}/alignment.json',gate)
        if not gate['passed']:raise AlignmentFailure('Native/TF disagreement before update; original tolerance unchanged')
        totals=prepare_targets(self.method,records)
        self.save(f'rollouts/batch{batch:03d}/training.json',{'records':records,'totals':totals,'batch_digest':digest(records)})
        return records,totals

    def update(self,records,totals,pass_index):
        self.optimize(records,totals,pass_index)
        if pass_index==2:self.completed_batches=self.nominal_slot//2

    def skip(self,slot):
        assert self.method=='rft' and slot==self.nominal_slot+1
        # The engine has already established N_positive==0; no optimizer/backward call.
        self.current_slot=self.nominal_slot=slot;self.completed_batches=slot//2
        self.counter['rft_skipped_slots']+=1;self.save_counters()

    def verify_loaded_adapter(self,role,adapter):
        from safetensors.torch import load_file
        payload=load_file(str(adapter/'adapter_model.safetensors'),device='cpu');actual=self.adapter_hashes(role)
        assert set(payload)=={n.replace('.default.','.') for n in actual}
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in payload.values())
        assert {n:tensor_hash(payload[n.replace('.default.','.')]) for n in actual}==actual

    def parameter_names(self):
        return {role:[name for name,p in self.models[role].named_parameters() if p.requires_grad]+(['head.'+name for name,p in self.head.named_parameters()] if role=='critic' else []) for role in self.parameters}

    def export(self,stage):
        from safetensors.torch import load_file
        for role in self.parameters:
            self.models[role].save_pretrained(stage/role,safe_serialization=True,save_embedding_layers=False)
            values=load_file(str(stage/role/'adapter_model.safetensors'),device='cpu');expected=self.adapter_hashes(role)
            assert set(values)=={n.replace('.default.','.') for n in expected}
            assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in values.values())
            assert {n:tensor_hash(values[n.replace('.default.','.')]) for n in expected}==expected
        if self.head is not None:torch.save(self.head.state_dict(),stage/'head.pt')
        state={'schema':CHECKPOINT,'spec_hash':self.run['spec_hash'],'roles':list(self.models),'steps':dict(self.steps),'nominal_slot':self.nominal_slot,'completed_rollouts':self.nominal_slot//2,'optimizers':{r:o.state_dict() for r,o in self.optimizers.items()},'trainable_names':self.parameter_names(),'rng':rng_state()}
        torch.save(state,stage/'state.pt');restored=torch.load(stage/'state.pt',map_location='cpu',weights_only=False)
        assert fingerprint(state)==fingerprint(restored)
        if self.head is not None:
            head=torch.load(stage/'head.pt',map_location='cpu',weights_only=True)
            assert fingerprint(head)==fingerprint(self.head.state_dict())
        self.check_frozen()
        return {'kind':'NATIVE_FM_V1','trainable_roles':list(self.parameters),'state_fingerprint':digest(fingerprint(state)),'adapters':{r:self.adapter_hashes(r) for r in self.models},'head':{n:tensor_hash(p) for n,p in self.head.named_parameters()} if self.head is not None else None,'state_roundtrip':'payload equality checked without a model probe'}

    def restore(self,resume,identity):
        state=torch.load(resume/'state.pt',map_location='cpu',weights_only=False)
        assert state['schema']==CHECKPOINT and state['spec_hash']==self.run['spec_hash'] and state['roles']==list(self.models)
        assert state['steps']==identity['actual_steps'] and state['nominal_slot']==identity['nominal_slot'] and state['completed_rollouts']==identity['completed_rollouts']
        assert digest(fingerprint(state))==identity['backend']['state_fingerprint']
        assert state['trainable_names']==self.parameter_names() and set(state['optimizers'])==set(self.optimizers)
        for role,optimizer in self.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][role]);verify_optimizer(optimizer,self.steps[role])
            if self.steps[role]>0:assert len(optimizer.state)==len(self.parameters[role]),'Missing saved optimizer slots'
        assert fingerprint({r:o.state_dict() for r,o in self.optimizers.items()})==fingerprint(state['optimizers'])
        assert {r:self.adapter_hashes(r) for r in self.models}==identity['backend']['adapters']
        assert ({n:tensor_hash(p) for n,p in self.head.named_parameters()} if self.head is not None else None)==identity['backend']['head']
        restore_rng(state['rng']);assert fingerprint(rng_state())==fingerprint(state['rng'])
        self.journal.add('restore_verified',checkpoint=resume.name,model_probes=0)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
