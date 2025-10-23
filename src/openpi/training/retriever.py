# 상단: import 근처
import numpy as np
import jax
import jax.numpy as jnp

import openpi.models.model as _model

# --- 간단 Retriever (내장) -----------------------------------------
class SimpleRetriever:
    def __init__(self, H, A, success_only=True, topk=8, alpha_q=0.3):
        self.H, self.A = H, A
        self.success_only = success_only
        self.topk = topk
        self.alpha_q = alpha_q  # sim-Q 가중 혼합
        self._keys = None      # [N,D]
        self._chunks = None    # [N,H,A]
        self._qhint = None     # [N]  (초기엔 reward 기반, 주기적으로 critic 스코어로 갱신)
    
    def build_from_buffer(self, model, buffer):
        # 버퍼의 모든 timestep에서 obs 임베딩(key)와 그 시점부터 H-step action chunk 추출
        keys, chunks, qhint = [], [], []
        for traj in buffer.iter_trajs():
            T = traj['episode_length']
            # suc = float(traj.get('is_success', False))
            R = traj['rewards']    # (T,)
            A = traj['actions']    # (T, orig or padded)
            # 관측을 expo 포맷으로 꺼내기 (buffer.sample 참조)
            base = traj['base_img']; wrist = traj['wrist_img']
            base_m, wrist_m = traj['base_img_mask'], traj['wrist_img_mask']
            state = traj['state']; tok = traj['tokenized_prompt']; tok_m = traj['tokenized_prompt_mask']
            # 관측 단일 step -> Observation 만들기
            for t in range(T):
                obs_dict = {
                  'image': {'base_0_rgb': jnp.array(base[t:t+1]), 'left_wrist_0_rgb': jnp.array(wrist[t:t+1]), 'right_wrist_0_rgb': jnp.zeros_like(jnp.array(base[t:t+1]))},
                  'image_mask': {'base_0_rgb': jnp.array(base_m[t:t+1]), 'left_wrist_0_rgb': jnp.array(wrist_m[t:t+1]), 'right_wrist_0_rgb': jnp.zeros_like(jnp.array(base_m[t:t+1]), dtype=bool)},
                  'state': jnp.array(state[t:t+1]),
                  'tokenized_prompt': jnp.array(tok[t:t+1]),
                  'tokenized_prompt_mask': jnp.array(tok_m[t:t+1]),
                }
                obs = _model.Observation.from_dict(obs_dict)
                # trunk embedding
                z = model.critic.base_cls.trunk(obs, train=False)  # [1,D]
                keys.append(np.asarray(z)[0])

                # H-step chunk
                h_end = min(t+self.H, T)
                chunk = A[t:h_end]
                if len(chunk) < self.H:
                    pad = np.tile(chunk[-1:], (self.H-len(chunk), 1))
                    chunk = np.concatenate([chunk, pad], axis=0)
                
                # Pad action from 7D to action_dim (keep original 7 dims; rest zeros)
                if self.A > chunk.shape[1]:
                    pad = np.zeros((self.H, self.A - chunk.shape[1]), dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, pad], axis=1)
                
                chunks.append(chunk)
                # 간단 Q 힌트: 향후 H 보상합
                qhint.append(R[t:h_end].sum())

        K = np.stack(keys, 0); C = np.stack(chunks, 0); Q = np.asarray(qhint)
        if self.success_only:
            mask = Q > 0.0
            if mask.any():
                K, C, Q = K[mask], C[mask], Q[mask]
        self._keys, self._chunks, self._qhint = K, C, Q

    def query_actions(self, model, obs_b):
        # obs_b: Observation (B,…) → [B,topk,H,A] 반환
        B = obs_b.state.shape[0]
        z = model.critic.base_cls.trunk(obs_b, train=False)         # [B,D]
        Z = np.asarray(z)                                           # CPU 연산
        K = self._keys; C = self._chunks; Q = self._qhint
        # L2 거리 → 유사도
        # sim = -||Z-K||^2
        sims = -( (Z[:,None,:]-K[None,:,:])**2 ).sum(axis=-1)       # [B,N]
        # 혼합 점수: alpha*sim + (1-alpha)*norm_Q
        Qn = (Q - Q.mean())/(Q.std()+1e-6)
        score = self.alpha_q*sims + (1-self.alpha_q)*Qn[None,:]
        # topk 인덱스
        idx = np.argpartition(-score, kth=min(self.topk, score.shape[1]-1), axis=1)[:, :self.topk]
        # 정렬
        row = np.arange(B)[:,None]
        ord = np.argsort(-score[row, idx], axis=1)
        top = idx[row, ord]                                         # [B,topk]
        acts = C[top]                                               # [B,topk,H,A]
        return jax.device_put(acts.astype(np.float32))
