import torch
import torch.nn as nn

from nflows.distributions import StandardNormal
from nflows.utils.torchutils import repeat_rows, split_leading_dim

from .odefunc import ODEfunc, ODEnet, divergence_bf
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


class ContinuousNormalizingFlow(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dims,
            context_dim=0,
            num_blocks=3,
            conditional=False,
            layer_type="concatsquash",
            nonlinearity="tanh",
            time_length=0.5,
            train_T=True,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            batch_norm=True,
            bn_lag=0.0,
            sync_bn=False,
            device=None
    ):
        super().__init__()
        self.device = device

        self.flow = self.build_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            context_dim=context_dim,
            num_blocks=num_blocks,
            conditional=conditional,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            time_length=time_length,
            train_T=train_T,
            solver=solver,
            atol=atol,
            rtol=rtol,
            use_adjoint=use_adjoint,
            batch_norm=batch_norm,
            bn_lag=bn_lag,
            sync_bn=sync_bn
        )
        self.flow = self.flow.to(self.device)
        self.distribution = StandardNormal(shape=[input_dim]).to(self.device)

    @staticmethod
    def build_model(
            input_dim,
            hidden_dims,
            context_dim=0,
            num_blocks=3,
            conditional=False,
            layer_type="concatsquash",
            nonlinearity="tanh",
            time_length=0.5,
            train_T=True,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            batch_norm=True,
            bn_lag=0.0,
            sync_bn=False
    ):
        def build_cnf():
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
            )
            odefunc = ODEfunc(
                diffeq=diffeq,
                divergence_fn=divergence_bf
            )
            cnf = CNF(
                odefunc=odefunc,
                T=time_length,
                train_T=train_T,
                conditional=conditional,
                solver=solver,
                use_adjoint=use_adjoint,
                atol=atol,
                rtol=rtol,
            )
            return cnf

        chain = [build_cnf() for _ in range(num_blocks)]
        if batch_norm:
            bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=bn_lag, sync=sync_bn) for _ in range(num_blocks)]
            bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=bn_lag, sync=sync_bn)]
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
            chain = bn_chain
        model = SequentialFlow(chain)
        return model

    def log_prob(self, X: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Calculate log probability on data (usually batch)."""
        zero: torch.Tensor = torch.zeros(X.shape[0], 1, device=X.device)
        z, delta_logp = self.flow(x=X, context=context, logpx=zero)

        logpz: torch.Tensor = self.distribution.log_prob(z)
        logpz: torch.Tensor = logpz.reshape(-1, 1)
        logpx: torch.Tensor = logpz - delta_logp
        return logpx

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample from the model (usually batch)."""
        context_shape = context.shape

        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=context.device)
        context: torch.Tensor = repeat_rows(context, num_reps=num_samples)
        base_dist_samples: torch.Tensor = self.distribution.sample(num_samples=context.shape[0])

        samples: torch.Tensor = self.flow(x=base_dist_samples, context=context, reverse=True)
        samples: torch.Tensor = split_leading_dim(samples, [context_shape[0], num_samples])
        return samples
