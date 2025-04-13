import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiModalAttentionExtractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 embed_dim: int = 128,
                 n_attn_heads: int = 4,
                 n_layers: int = 2):
        super().__init__(observation_space, features_dim=embed_dim)
        self.embed_dim = embed_dim

        # 각 모달리티에 대한 선형 임베딩 레이어 정의
        self.nav_embed = nn.Linear(20 * 12, embed_dim)  # 20x12 평탄화 -> embed_dim
        self.agent_embed = nn.Linear(21 * 11,
                                     embed_dim)  # 21x11 평탄화 -> embed_dim
        self.lane_embed = nn.Linear(20 * 12,
                                    embed_dim)  # 20x12 평탄화 -> embed_dim
        self.static_embed = nn.Linear(10, embed_dim)  # 10 -> embed_dim

        # 위치 인코딩 (선택 사항): 가능한 각 위치에 대해 단순 학습 파라미터 사용
        self.nav_pos_emb = nn.Parameter(th.zeros(25, embed_dim))
        self.agent_pos_emb = nn.Parameter(th.zeros(32, embed_dim))
        self.lane_pos_emb = nn.Parameter(th.zeros(70, embed_dim))
        self.static_pos_emb = nn.Parameter(th.zeros(5, embed_dim))

        # 모달리티 타입 임베딩 (선택 사항)
        self.type_emb = nn.Parameter(th.zeros(4, embed_dim))  # 각 모달리티별 하나씩

        # Transformer encoder 레이어
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=n_attn_heads,
                                                   dim_feedforward=embed_dim *
                                                   4,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=n_layers)

    def forward(self, observations: dict) -> th.Tensor:
        # 각 모달리티 추출 및 평탄화
        nav = observations['nav'].reshape(observations['nav'].shape[0], 25,
                                          -1)  # (B,25,240)
        agents = observations['agents'].reshape(observations['agents'].shape[0],
                                                32, -1)  # (B,32,231)
        lane = observations['lane'].reshape(observations['lane'].shape[0], 70,
                                            -1)  # (B,70,240)
        static = observations['static'].reshape(observations['static'].shape[0],
                                                5, -1)  # (B,5,10)

        # 선형 임베딩 적용
        nav_tokens = self.nav_embed(nav)  # (B,25,D)
        agent_tokens = self.agent_embed(agents)  # (B,32,D)
        lane_tokens = self.lane_embed(lane)  # (B,70,D)
        static_tokens = self.static_embed(static)  # (B,5,D)

        # 위치 인코딩 추가
        nav_tokens = nav_tokens + self.nav_pos_emb  # (25,D)가 각 배치에 브로드캐스트됨
        agent_tokens = agent_tokens + self.agent_pos_emb
        lane_tokens = lane_tokens + self.lane_pos_emb
        static_tokens = static_tokens + self.static_pos_emb

        # 모달리티 타입 인코딩 추가
        nav_tokens = nav_tokens + self.type_emb[
            0]  # type_emb[0]은 (D,)이며, 모든 nav 토큰에 더함
        agent_tokens = agent_tokens + self.type_emb[1]
        lane_tokens = lane_tokens + self.type_emb[2]
        static_tokens = static_tokens + self.type_emb[3]

        # 모든 토큰 연결
        combined_tokens = th.cat(
            [nav_tokens, agent_tokens, lane_tokens, static_tokens],
            dim=1)  # (B, 132, D)

        # 어텐션 key_padding_mask 생성: shape (B, 132), 패딩인 경우 True
        nav_mask = observations['nav_mask']  # (B,25)
        agent_mask = observations['agents_mask']  # (B,32)
        lane_mask = observations['lane_mask']  # (B,70)
        static_mask = observations['static_mask']  # (B,5)
        # 모든 토큰에 대한 마스크 연결
        combined_mask = th.cat([nav_mask, agent_mask, lane_mask, static_mask],
                               dim=1)  # (B,132)
        # PyTorch Transformer는 패딩된 항목에 대해 True를 기대하므로 마스크 반전
        key_padding_mask = combined_mask == 0

        # Transformer encoder (self-attention 레이어) 적용
        # key_padding_mask는 패딩 토큰에 대한 어텐션을 방지함
        encoded_tokens = self.transformer_encoder(
            combined_tokens, src_key_padding_mask=key_padding_mask)  # (B,132,D)

        # 토큰 특징 집계: masked average pooling
        # 평균을 내기 전에 패딩된 토큰은 0으로 처리
        mask = combined_mask.unsqueeze(-1)  # (B,132,1)
        # 분모에 최소 1을 사용하여 0으로 나누는 것을 방지
        valid_counts = combined_mask.sum(dim=1,
                                         keepdim=True).clamp(min=1).unsqueeze(
                                             -1)  # (B,1,1)
        pooled = (encoded_tokens * mask).sum(
            dim=1, keepdim=True) / valid_counts  # (B,1,D)
        # 시퀀스 차원 제거
        features = pooled.squeeze(1)  # (B, D)

        return features  # 결과는 (B, embed_dim)의 크기를 가짐
