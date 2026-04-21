import torch
from torch import nn as nn

class TokenSelect(nn.Module):
    def __init__(
        self,
        expansion_step: list = [0, 100, 200],  # 扩展阶段的步数，决定在哪些epoch进行token的扩展
        keep_rate: list = [0.5, 0.75, 1.0],  # 在每个扩展阶段保留的token的比例
        initialization_keep_rate: float = 0.25,  # 初始阶段保留的token比例
        expansion_multiple_stage: int = 2,  # 每个扩展阶段内部的多重扩展次数
        distance: str = "cosine",  # 使用的距离度量方法（余弦距离、曼哈顿距离、欧几里得距离）
    ):
        super().__init__()
        self.expansion_stage = 0  # 当前扩展阶段，初始为0
        self.sparse_inference = False  # 是否进行稀疏推理

        self.expansion_step = expansion_step  # 保存扩展步数
        self.total_expansion_stage = len(expansion_step)  # 总共的扩展阶段数
        self.initialization_keep_rate = initialization_keep_rate  # 保存初始保留比例

        self.expansion_keep_rate = []  # 初始化扩展保留比例列表
        for i in range(len(keep_rate)):
            if i == 0:
                self.expansion_keep_rate.append(keep_rate[i] - initialization_keep_rate)  
                # 第一个阶段的扩展保留比例为keep_rate减去初始保留比例
            else:
                self.expansion_keep_rate.append(keep_rate[i] - keep_rate[i - 1])
                # 后续阶段的扩展保留比例为当前阶段keep_rate减去前一阶段的keep_rate

        self.final_keep_rate = keep_rate[-1]  # 最终保留比例为最后一个keep_rate
        self.expansion_multiple_stage = expansion_multiple_stage  # 多重扩展次数

        self.distance = distance  # 距离度量方法

    def update_current_stage(self, epoch: int):
        import bisect

        expansion_stage = bisect.bisect_right(self.expansion_step, epoch)
        # 使用二分查找算法，确定当前epoch对应的扩展阶段
        self.expansion_stage = expansion_stage  # 更新当前扩展阶段

    def get_score(self, a: torch.Tensor, b: torch.Tensor):
        if self.distance == "cosine":
            dist = a @ b.transpose(-1, -2)  # 计算余弦相似度
        elif self.distance == "manhattan":
            dist = torch.sum(
                torch.abs(a.unsqueeze(2) - b.unsqueeze(1)),
                dim=-1,
            )  # 计算曼哈顿距离
        elif self.distance == "euclidean":
            dist = torch.sqrt(torch.sum((a.unsqueeze(2) - b.unsqueeze(1)) ** 2, dim=-1))
            # 计算欧几里得距离
        else:
            raise Exception("Wrong distance!", self.distance)
        return dist  # 返回计算的距离或相似度

    def token_initialization(self, token: torch.Tensor):
        x = int((self.token_num - 1) * self.initialization_keep_rate)
        # 计算初始保留的token数量
        step = int(1 // self.initialization_keep_rate)  # 计算采样步长
        with torch.no_grad():
            select_index = []  # 用于保存选择的token索引
            unselect_index = []  # 用于保存未选择的token索引
            for i in range(self.token_num - 1):
                if i % step == 0 and len(select_index) < x:
                    select_index.append(i)  # 按照步长选择token
                else:
                    unselect_index.append(i)  # 其余token归为未选择
            select_index = (
                torch.tensor(select_index)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device=token.device)
            ).expand(
                token.shape[0],
                x,
                token.shape[2],
            )  # 扩展选中的索引，以便与token进行对齐
            unselect_index = (
                torch.tensor(unselect_index)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device=token.device)
            ).expand(
                token.shape[0],
                token.shape[1] - x,
                token.shape[2],
            )  # 扩展未选中的索引

        select_token = token.gather(dim=1, index=select_index)
        # 根据索引选取对应的token
        unselect_token = token.gather(dim=1, index=unselect_index)
        # 选取未选择的token

        assert select_token.shape[1] + unselect_token.shape[1] == (
            self.token_num - 1
        ), "Wrong shape!"  # 确保选择和未选择的token数量正确
        assert select_index.shape[1] + unselect_index.shape[1] == (
            self.token_num - 1
        ), "Wrong shape!"  # 确保选择和未选择的索引数量正确

        return (select_token, select_index), (unselect_token, unselect_index)
        # 返回选择的token和索引，以及未选择的token和索引

    def token_expansion(
        self,
        select_token: torch.Tensor,
        select_index: torch.Tensor,
        unselect_token: torch.Tensor,
        unselect_index: torch.Tensor,
    ):
        for stage in range(1, self.expansion_stage + 1):  # 遍历当前扩展阶段
            if stage == self.total_expansion_stage:
                expansion_token_num = int(
                    (self.token_num - 1) * self.final_keep_rate
                ) - int(
                    (self.token_num - 1)
                    * (
                        self.initialization_keep_rate
                        + sum([self.expansion_keep_rate[i] for i in range(stage - 1)])
                    )
                )  # 计算在最后阶段保留的token数量
            else:
                expansion_token_num = int(
                    (self.token_num - 1) * self.expansion_keep_rate[stage - 1]
                )  # 计算在非最后阶段保留的token数量

            for k in range(1, self.expansion_multiple_stage + 1):
                if k == self.expansion_multiple_stage:
                    multiple_expansion_token_num = expansion_token_num - (
                        self.expansion_multiple_stage - 1
                    ) * (expansion_token_num // self.expansion_multiple_stage)
                else:
                    multiple_expansion_token_num = (
                        expansion_token_num // self.expansion_multiple_stage
                    )
                # 按照多重扩展次数，计算每次扩展的token数量

                with torch.no_grad():
                    select_token_norm = select_token / select_token.norm(
                        dim=-1, keepdim=True
                    )  # 对选择的token进行归一化
                    unselect_token_norm = unselect_token / unselect_token.norm(
                        dim=-1, keepdim=True
                    )  # 对未选择的token进行归一化

                    scores = self.get_score(unselect_token_norm, select_token_norm)
                    # 计算选择token和未选择token之间的相似度

                    node_max, node_idx = scores.max(dim=-1)  # 获取最大相似度及其索引
                    edge_idx = node_max.argsort(dim=-1, descending=False)[..., None]
                    # 按照相似度对未选择token排序

                    add_node_index = edge_idx[..., :multiple_expansion_token_num, :]
                    # 选择需要扩展的token索引
                    unadd_node_index = edge_idx[..., multiple_expansion_token_num:, :]
                    # 剩下的为未选择的token索引

                add_index = unselect_index.gather(
                    dim=1,
                    index=add_node_index.expand(
                        unselect_token.shape[0],
                        multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )  # 获取扩展后的token索引
                add_token = unselect_token.gather(
                    dim=1,
                    index=add_node_index.expand(
                        unselect_token.shape[0],
                        multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )  # 获取扩展后的token
                select_index = torch.cat([select_index, add_index], dim=1)
                # 将扩展后的索引加入已选择索引
                select_token = torch.cat([select_token, add_token], dim=1)
                # 将扩展后的token加入已选择token

                unselect_index = unselect_index.gather(
                    dim=1,
                    index=unadd_node_index.expand(
                        unselect_token.shape[0],
                        unselect_token.shape[1] - multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )  # 更新未选择的索引
                unselect_token = unselect_token.gather(
                    dim=1,
                    index=unadd_node_index.expand(
                        unselect_token.shape[0],
                        unselect_token.shape[1] - multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )  # 更新未选择的token

                assert select_token.shape[1] + unselect_token.shape[1] == (
                    self.token_num - 1
                ), "Wrong shape!"  # 确保扩展后的token数量正确
                assert select_index.shape[1] + unselect_index.shape[1] == (
                    self.token_num - 1
                ), "Wrong shape!"  # 确保扩展后的索引数量正确
        return (select_token, select_index), (unselect_token, unselect_index)
        # 返回扩展后的选择token和索引，以及未选择的token和索引

    def token_merge(
        self,
        select_token: torch.Tensor,
        select_index: torch.Tensor,
        unselect_token: torch.Tensor,
        unselect_index: torch.Tensor,
        mode="mean",
    ):
        rest_token_num = unselect_token.shape[1]  # 获取剩余的未选择token数量

        with torch.no_grad():
            select_token_norm = select_token / select_token.norm(dim=-1, keepdim=True)
            # 对选择的token进行归一化
            unselect_token_norm = unselect_token / unselect_token.norm(
                dim=-1, keepdim=True
            )
            # 对未选择的token进行归一化
            scores = self.get_score(unselect_token_norm, select_token_norm)
            # 计算选择token和未选择token之间的相似度

            node_max, node_idx = scores.max(dim=-1)  # 获取最大相似度及其索引
            merge_unselect_node_index = node_idx[..., None]
            # 选择相似度最高的索引，用于后续的合并

        select_token = select_token.scatter_reduce(
            dim=1,
            index=merge_unselect_node_index.expand(
                unselect_token.shape[0],
                rest_token_num,
                unselect_token.shape[2],
            ),
            src=unselect_token,
            reduce=mode,
        )
        # 根据相似度将未选择的token与选择的token合并

        return (select_token, select_index)  # 返回合并后的token和索引

    def token_select(self, x):
        self.token_num = x.shape[1]  # 获取输入token的数量
        select_index = None  # 初始化选择索引
        if (
            self.expansion_stage > 0
            and not (
                self.expansion_stage == self.total_expansion_stage
                and self.final_keep_rate == 1.0
            )
            and self.sparse_inference
        ):
            # 如果处于扩展阶段且稀疏推理开启，则进行token选择
            token_cls = x[..., :1, :]  # 分离CLS token
            (select_token, select_index), (
                unselect_token,
                unselect_index,
            ) = self.token_initialization(x[..., 1:, :])
            # 对输入token（除CLS外）进行初始化选择
            (select_token, select_index), (
                unselect_token,
                unselect_index,
            ) = self.token_expansion(
                select_token,
                select_index,
                unselect_token,
                unselect_index,
            )
            # 对选择后的token进行扩展
            if unselect_token.shape[1] > 0:
                (select_token, select_index) = self.token_merge(
                    select_token, select_index, unselect_token, unselect_index, "mean"
                )
                # 如果仍有未选择的token，则进行合并

            x = torch.cat([token_cls, select_token], dim=1)
            # 将CLS token与选择的token合并
            cls_index = torch.zeros([x.shape[0], 1, x.shape[2]]).to(
                device=select_index.device
            )  # 初始化CLS索引
            select_index = select_index + 1  # 更新选择索引
            select_index = torch.cat([cls_index, select_index], dim=1)
            # 将CLS索引与选择索引合并
            select_index = select_index.long()  # 转换索引为长整型
            assert x.shape[1] == select_index.shape[1], "Wrong shape!"
            # 确保token和索引的数量匹配

        return x, select_index  # 返回选择后的token和索引

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    token_selector = TokenSelect(
        expansion_step=[0, 50, 100],  # 扩展步骤，例如在epoch 0, 50, 100进行扩展
        keep_rate=[0.5, 0.75, 1.0],   # 每个阶段保留的token比例
        initialization_keep_rate=0.25, # 初始保留比例
        expansion_multiple_stage=2,   # 每个阶段的扩展次数
        distance="cosine"              # 使用的距离度量方法
    )
    input_tensor = torch.rand(1, 2200, 512)
    # 假设当前处于第20个epoch
    current_epoch = 51
    token_selector.update_current_stage(current_epoch)

    # 调用token_select函数，选择token并返回选择的索引
    selected_tokens, selected_indices = token_selector(input_tensor)

    # 输出结果
    print("Selected Tokens Shape:", selected_tokens.shape)
    print("Selected Indices Shape:", selected_indices.shape)