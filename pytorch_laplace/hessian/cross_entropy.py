from typing import Optional, Tuple

import nnj
import torch
from backpack import extend
from torch import nn

from pytorch_laplace.hessian.base import HessianCalculator


class CEHessianCalculator(HessianCalculator):
    """
    Multi-Class Cross Entropy

    .. warning::
        Currently only support one point prediction (for now)
        for example:
            - Classifying mnist digits will work
            - Pixelwise classification to segment digits will not work
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lossfunc = extend(nn.CrossEntropyLoss())

    @torch.no_grad()
    def _compute_hessian_nnj(
        self,
        x: torch.Tensor,
        model: nnj.Sequential,
        target: Optional[torch.Tensor] = None,
        save_memory: bool = False,
        reshape: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Compute Hessian of cross entropy

        Args:
            x: input of the network
            target: output of the network (not used for cross entropy)
            model: neural network module
            reshape: reshape logits to this shape before computing cross entropy
        """

        val = model(x)
        if reshape is not None:
            val = val.reshape(val.shape[0], *reshape)

        if len(val.shape) == 2:
            # single point classification

            exp_val = torch.exp(val)
            softmax = torch.einsum("bi,b->bi", exp_val, 1.0 / torch.sum(exp_val, dim=1))

            # hessian = diag(softmax) - softmax.T * softmax
            # thus Jt * hessian * J = Jt * diag(softmax) * J - Jt * softmax.T * softmax * J

            # backpropagate through the network the diagonal part
            Jt_diag_J = model.jTmjp(
                x,
                val,
                softmax,
                wrt="weight",
                from_diag=True,
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            # backpropagate through the network the outer product
            softmax_J = model.vjp(x, val, softmax, wrt="weight")
            if self.hessian_shape == "diag":
                Jt_outer_J = torch.einsum("bi,bi->bi", softmax_J, softmax_J)
            else:
                Jt_outer_J = torch.einsum("bi,bj->bij", softmax_J, softmax_J)

            # add the backpropagated quantities
            Jt_H_J = Jt_diag_J - Jt_outer_J

            # average along batch size
            Jt_H_J = torch.mean(Jt_H_J, dim=0)
            return Jt_H_J

        if len(val.shape) == 3:
            # multi point classification

            b, p, c = val.shape

            exp_val = torch.exp(val)
            softmax = torch.einsum("bpi,bp->bpi", exp_val, 1.0 / torch.sum(exp_val, dim=2))

            # backpropagate through the network the diagonal part
            diagonal = softmax.reshape(val.shape[0], val.shape[1:].numel())
            Jt_diag_J = model.jTmjp(
                x,
                val,
                diagonal,
                wrt="weight",
                from_diag=True,
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            # backpropagate through the network the outer product
            if save_memory is True:
                Jt_outer_J = torch.zeros_like(Jt_diag_J)
                for point in range(p):
                    vector = torch.zeros(b, p * c, device=val.device)
                    vector[:, point * c : (point + 1) * c] = softmax[:, point, :]
                    softmax_J = model.vjp(x, val, vector, wrt="weight")
                    if self.hessian_shape == "diag":
                        Jt_outer_J += torch.einsum("bi,bi->bi", softmax_J, softmax_J)
                    else:
                        Jt_outer_J += torch.einsum("bi,bj->bij", softmax_J, softmax_J)
            elif save_memory is False:
                pos_identity = torch.diag_embed(torch.ones(p, device=val.device))
                matrix = torch.einsum("bpi,pq->bpqi", softmax, pos_identity).reshape(b, p, p * c)
                softmax_J = model.mjp(x, val, matrix, wrt="weight")
                if self.hessian_shape == "diag":
                    Jt_outer_J = torch.einsum("bki,bki->bi", softmax_J, softmax_J)
                else:
                    Jt_outer_J = torch.einsum("bki,bkj->bij", softmax_J, softmax_J)
            else:
                Jt_outer_J = torch.zeros_like(Jt_diag_J)
                batch_size = int(p / save_memory)
                assert batch_size == p / save_memory
                pos_identity = torch.diag_embed(torch.ones(batch_size, device=val.device))
                for batch_n in range(save_memory):
                    matrix = torch.einsum(
                        "bpi,pq->bpqi",
                        softmax[:, batch_n * batch_size : (batch_n + 1) * batch_size, :],
                        pos_identity,
                    ).reshape(b, batch_size, batch_size * c)
                    matrix = torch.cat(
                        [
                            torch.zeros(b, batch_size, (batch_n * batch_size) * c, device=val.device),
                            matrix,
                            torch.zeros(
                                b,
                                batch_size,
                                ((save_memory - batch_n - 1) * batch_size) * c,
                                device=val.device,
                            ),
                        ],
                        dim=2,
                    )
                    softmax_J = model.mjp(x, val, matrix, wrt="weight")
                    if self.hessian_shape == "diag":
                        Jt_outer_J += torch.einsum("bki,bki->bi", softmax_J, softmax_J)
                    else:
                        Jt_outer_J += torch.einsum("bki,bkj->bij", softmax_J, softmax_J)

            # add the backpropagated quantities
            Jt_H_J = Jt_diag_J - Jt_outer_J

            # average along batch size
            Jt_H_J = torch.mean(Jt_H_J, dim=0)
            return Jt_H_J
