import time

import torch

from utils import *

from GDCMAD.earlyStopping import EarlyStopping
from GDCMAD.graph import *

import torch.nn.functional as F
import math

device = get_default_device()

T = torch.Tensor


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight_1 = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_weight_2 = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight_1, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.base_weight_2, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight_1)
        spline_output = F.linear(
            torch.nn.LeakyReLU(negative_slope=0.01)(x),
            self.base_weight_2,
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KANEfficient(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.kan = KANLinear(
            in_features=in_features,
            out_features=out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )

    def forward(self, x: T) -> T:
        batch_size, seq_length, _ = x.shape
        x = self.kan(x)
        return x.view(batch_size, seq_length, -1)


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=3, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = KANEfficient(model_dim, self.dim_per_head * num_heads)
        self.linear_v = KANEfficient(model_dim, self.dim_per_head * num_heads)
        self.linear_q = KANEfficient(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = KANEfficient(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        #
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        #
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            #
            attention = attention.masked_fill_(attn_mask, -np.inf)
        #
        attention = self.softmax(attention)
        #
        attention = self.dropout(attention)
        #
        context = torch.bmm(attention, v)
        return context, attention


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim, ffn_dim, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.permute(0, 2, 1)
        output = self.w2(F.relu(self.w1(output)))
        output = output.permute(0, 2, 1)
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        return output


class Attention_layer(nn.Module):
    def __init__(self, n_feature, num_heads, hid_dim, dropout=0.1):
        super(Attention_layer, self).__init__()

        #
        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(n_feature, hid_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        output = self.feed_forward(context)

        return output, attention


class GDCMADModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.device = opt.device

        self.num_levels = 1

        self.loss_cl_enc_ratio = opt.loss_cl_enc_ratio
        self.loss_cl_ratio = opt.loss_cl_ratio

        self.ver_module = GraphEmbedding(num_nodes=opt.dim, seq_len=opt.window_size, num_levels=self.num_levels,
                                         device=torch.device(device))

        self.hor_module = GraphEmbedding(num_nodes=opt.window_size, seq_len=opt.dim, num_levels=self.num_levels,
                                         device=torch.device(device))

        self.ver_relu = nn.ReLU()
        self.hor_relu = nn.ReLU()

        self.ver_atte = Attention_layer(n_feature=opt.window_size, num_heads=1, hid_dim=opt.window_size,
                                        dropout=opt.drop_out)
        self.hor_atte = Attention_layer(n_feature=opt.dim, num_heads=1, hid_dim=opt.dim,
                                        dropout=opt.drop_out)

        self.final_out_channels = opt.window_size
        self.features_len = opt.dim
        self.project_channels = opt.project_channels
        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // 2),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // 2, self.project_channels),
        )

        self.fusion = nn.Linear(self.project_channels * 2, self.project_channels)

        self.decoder = Attention_layer(n_feature=self.project_channels, num_heads=1, hid_dim=1,
                                       dropout=opt.drop_out)

        self.projection_dec = nn.Sequential(
            nn.Linear(self.project_channels, self.final_out_channels * self.features_len // 2),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // 2, self.final_out_channels * self.features_len),
        )

        self.dropout = nn.Dropout(opt.drop_out)
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.LSTM(opt.dim, opt.lstm_h_dim)
        self.encoder1 = nn.LSTM(opt.dim, opt.lstm_h_dim)
        self.encoder2 = nn.LSTM(opt.dim, opt.lstm_h_dim)
        self.projection_head_enc = nn.Sequential(
            nn.Linear(self.final_out_channels * opt.lstm_h_dim, self.final_out_channels * opt.lstm_h_dim // 2),
            nn.BatchNorm1d(self.final_out_channels * opt.lstm_h_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * opt.lstm_h_dim // 2, self.project_channels),
        )

    def training_step(self, x):
        batch_size = x.size(0)
        win_size = x.size(1)
        fea_size = x.size(2)

        input = x
        #
        ver_input = input
        hor_input = input

        vertical_outputs = self.ver_module(ver_input)
        vertical_outputs = self.ver_relu(vertical_outputs)
        vertical_outputs = vertical_outputs.transpose(1, 2)
        vertical_outputs, _ = self.ver_atte(vertical_outputs)
        vertical_outputs = vertical_outputs.transpose(1, 2)
        vertical_outputs = vertical_outputs.reshape(vertical_outputs.size(0), -1)
        vertical_outputs = self.projection_head(vertical_outputs)

        horizontal_outputs = self.hor_module(hor_input.permute(0, 2, 1))
        horizontal_outputs = horizontal_outputs.transpose(1, 2)
        horizontal_outputs = self.hor_relu(horizontal_outputs)
        horizontal_outputs, _ = self.hor_atte(horizontal_outputs)
        horizontal_outputs = horizontal_outputs.reshape(horizontal_outputs.size(0), -1)
        horizontal_outputs = self.projection_head(horizontal_outputs)

        loss_cl = 1 - F.cosine_similarity(vertical_outputs, horizontal_outputs, eps=1e-6)
        loss_cl = torch.mean(loss_cl)

        hid_var = torch.cat([vertical_outputs, horizontal_outputs], dim=-1)
        hid_var = self.fusion(hid_var)
        hid_var = hid_var.unsqueeze(1)

        output, _ = self.decoder(hid_var)
        output = output.squeeze(1)
        output = self.projection_dec(output)

        output = output.reshape(-1, win_size, fea_size)
        output = self.dropout(output)
        output = self.sigmoid(output)

        loss_rec = torch.mean((x - output) ** 2)

        #
        raw_enc_outputs, _ = self.encoder1(x)
        raw_enc_outputs = raw_enc_outputs.reshape(raw_enc_outputs.size(0), -1)
        raw_enc_outputs = self.projection_head_enc(raw_enc_outputs)

        rec_enc_outputs, _ = self.encoder2(output)
        rec_enc_outputs = rec_enc_outputs.reshape(rec_enc_outputs.size(0), -1)
        rec_enc_outputs = self.projection_head_enc(rec_enc_outputs)

        loss_cl_enc = 1 - F.cosine_similarity(raw_enc_outputs, rec_enc_outputs, eps=1e-6)
        loss_cl_enc = torch.mean(loss_cl_enc)

        loss = self.loss_cl_enc_ratio * loss_cl_enc + self.loss_cl_ratio * loss_cl + loss_rec

        return loss

    def validation_step(self, x):
        batch_size = x.size(0)
        win_size = x.size(1)
        fea_size = x.size(2)

        input = x
        #
        ver_input = input
        hor_input = input

        vertical_outputs = self.ver_module(ver_input)
        vertical_outputs = self.ver_relu(vertical_outputs)
        vertical_outputs = vertical_outputs.transpose(1, 2)
        vertical_outputs, _ = self.ver_atte(vertical_outputs)
        vertical_outputs = vertical_outputs.transpose(1, 2)
        vertical_outputs = vertical_outputs.reshape(vertical_outputs.size(0), -1)
        vertical_outputs = self.projection_head(vertical_outputs)

        horizontal_outputs = self.hor_module(hor_input.permute(0, 2, 1))
        horizontal_outputs = horizontal_outputs.transpose(1, 2)
        horizontal_outputs = self.hor_relu(horizontal_outputs)
        horizontal_outputs, _ = self.hor_atte(horizontal_outputs)
        horizontal_outputs = horizontal_outputs.reshape(horizontal_outputs.size(0), -1)
        horizontal_outputs = self.projection_head(horizontal_outputs)

        hid_var = torch.cat([vertical_outputs, horizontal_outputs], dim=-1)
        hid_var = self.fusion(hid_var)
        hid_var = hid_var.unsqueeze(1)

        output, _ = self.decoder(hid_var)
        output = output.squeeze(1)
        output = self.projection_dec(output)

        output = output.reshape(-1, win_size, fea_size)
        output = self.dropout(output)
        output = self.sigmoid(output)

        score = torch.mean(torch.mean((x - output) ** 2, axis=2), axis=1)
        loss = torch.mean(score)
        return loss, score

    def validation_epoch_end(self, outputs):
        batch_losses = [x for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return epoch_loss

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss: {:.4f}".format(epoch, result))


def evaluate(model, val_loader):
    outputs = []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.to(torch.float32)
            data = to_device(data, device)
            #
            output, score = model.validation_step(data)
            outputs.append(output)

    return model.validation_epoch_end(outputs), score


def training(opt, model, train_loader, val_loader, model_path, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters())

    early_stopping = EarlyStopping(opt, patience=opt.patience, verbose=False, model_path=model_path)

    cur_epoch = 0

    train_steps = len(train_loader)

    start_time = time.time()

    for epoch in range(opt.niter):
        model.train()
        cur_epoch += 1

        epoch_start_time = time.time()
        epoch_train_time = []
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            if data.size(0) == 1:
                continue
            #
            optimizer.zero_grad()

            epoch_iter += 1
            data = data.to(torch.float32)
            data = to_device(data, device)

            #
            loss = model.training_step(data)

            loss.backward()
            optimizer.step()

        #
        epoch_train_time.append(time.time() - epoch_start_time)

        val_error, score = evaluate(model, val_loader)
        model.epoch_end(epoch, val_error)

        early_stopping(val_error, model)

        print('epoch', epoch)

        if early_stopping.early_stop:
            print('train finished with early stopping')
            break

        history.append(val_error)

    if not early_stopping.early_stop:
        print('train finished with total epochs')
        torch.save(model.state_dict(), model_path)

    total_train_time = time.time() - start_time

    return total_train_time, history, np.mean(epoch_train_time)


def testing(model, test_loader, alpha=.5, beta=.5):
    results = []
    pred_time = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            start_time = time.time()

            batch = batch.to(torch.float32)
            batch = to_device(batch, device)

            loss, score = model.validation_step(batch)

            results.append(score)

            pred_time.append(time.time() - start_time)

    return results, np.mean(pred_time)
