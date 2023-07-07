from typing import Literal, Optional, Tuple, Union

import nnj
import torch

from pytorch_laplace.hessian.base import HessianCalculator


def _arccos(z1: torch.Tensor, z2: torch.Tensor):
    """
    TODO: docstring
    """
    z1_norm = torch.sum(z1**2, dim=1) ** (0.5)
    z2_norm = torch.sum(z2**2, dim=1) ** (0.5)
    return 0.5 * torch.einsum("bi,bi->b", z1, z2) / (z1_norm * z2_norm)


def _arccos_hessian(z1: torch.Tensor, z2: torch.Tensor):
    """
    TODO: docstring
    """
    z1_norm = torch.sum(z1**2, dim=1) ** (0.5)
    z2_norm = torch.sum(z2**2, dim=1) ** (0.5)
    z1_normalized = torch.einsum("bi,b->bi", z1, 1 / z1_norm)
    z2_normalized = torch.einsum("bi,b->bi", z2, 1 / z2_norm)
    cosine = torch.einsum("bi,bi->b", z1_normalized, z2_normalized)

    zs = z1.shape
    identity = torch.eye(zs[1], dtype=z1.dtype, device=z1.device).repeat(zs[0], 1, 1)
    cosine_times_identity = torch.einsum("bij,b->bij", identity, cosine)

    outer_11 = torch.einsum("bi,bj->bij", z1_normalized, z1_normalized)
    outer_12 = torch.einsum("bi,bj->bij", z1_normalized, z2_normalized)
    outer_21 = torch.einsum("bi,bj->bij", z2_normalized, z1_normalized)
    outer_22 = torch.einsum("bi,bj->bij", z2_normalized, z2_normalized)

    cosine_times_outer_11 = torch.einsum("bij,b->bij", outer_11, cosine)
    cosine_times_outer_21 = torch.einsum("bij,b->bij", outer_21, cosine)
    cosine_times_outer_22 = torch.einsum("bij,b->bij", outer_22, cosine)

    H_11 = cosine_times_identity + outer_12 + outer_21 - 3 * cosine_times_outer_11
    H_12 = -identity + outer_11 + outer_22 - cosine_times_outer_21
    H_22 = cosine_times_identity + outer_12 + outer_21 - 3 * cosine_times_outer_22

    H_11_normalized = torch.einsum("bij,b->bij", H_11, 1 / (z1_norm**2))
    H_12_normalized = torch.einsum("bij,b->bij", H_12, 1 / (z1_norm * z2_norm))
    H_22_normalized = torch.einsum("bij,b->bij", H_22, 1 / (z2_norm**2))
    return H_11_normalized, H_12_normalized, H_22_normalized


class ContrastiveHessianCalculator(HessianCalculator):
    """
    Contrastive Loss

    .. math::
        L(x,y) = 0.5 * || x - y ||

    .. math::
        Contrastive(x, tuples) = \sum_{positives} L(x,y) - \sum_{negatives} L(x,y)

    .. note::
        The contrastive loss value is the same for method == "full" and method == "fix", however,
        the second order derivatives vary.
    """

    def __init__(self, method: Literal["full", "fix", "pos"], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.method = method
        assert self.method in ("full", "fix", "pos")

    @torch.no_grad()
    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
        tuple_indices: Tuple,
    ) -> torch.Tensor:
        """
        Compute contrastive loss

        .. math::
            L_{con} = 0.5 * || x - y ||^2

        where :math:`x` and :math:`y` are the embeddings of the anchor and positive/negative samples.

        Args:
            x: Images of the anchor and positive/negative samples.
            target: Embeddings of the anchor and positive/negative samples.
            model: Neural network module.
            tuple_indices: Tuple indices, either (a,p,n) or (a,p,a,n).
        """
        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        # compute positive part
        pos = 0.5 * (model(x[ap]) - model(x[p])) ** 2

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.method == "pos":
            return pos

        # compute negative part
        neg = 0.5 * (model(x[an]) - model(x[n])) ** 2

        # sum along batch size
        neg = torch.sum(neg, dim=0)

        return pos - neg

    @torch.no_grad()
    def compute_gradient(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
        tuple_indices: Tuple,
    ) -> torch.Tensor:
        """
        Compute contrastive gradient

        Args:
            x: Images of the anchor and positive/negative samples.
            target: Embeddings of the anchor and positive/negative samples.
            model: Neural network module.
            tuple_indices: Tuple indices, either (a,p,n) or (a,p,a,n).
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_hessian(
        self,
        x: torch.Tensor,
        model: nnj.Sequential,
        tuple_indices: Tuple,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive hessian

        Args:
            x: Images of the anchor and positive/negative samples.
            target: Embeddings of the anchor and positive/negative samples.
            model: Neural network module.
            tuple_indices: Tuple indices, either (a,p,n) or (a,p,a,n).
        """

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        if self.method == "full" or self.method == "pos":
            # compute positive part
            pos = model.jTmjp_batch2(
                x[ap],
                x[p],
                None,
                None,
                None,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            if self.hessian_shape == "diag":
                pos = pos[0] - 2 * pos[1] + pos[2]
            else:
                raise NotImplementedError
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            if self.method == "pos":
                return pos

            # compute negative part
            neg = model.jTmjp_batch2(
                x[an],
                x[n],
                None,
                None,
                None,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            if self.hessian_shape == "diag":
                neg = neg[0] - 2 * neg[1] + neg[2]
            else:
                raise NotImplementedError
            # sum along batch size
            neg = torch.sum(neg, dim=0)

            return pos - neg

        if self.method == "fix":
            positives = x[p] if len(tuple_indices) == 3 else torch.cat((x[ap], x[p]))
            negatives = x[n] if len(tuple_indices) == 3 else torch.cat((x[an], x[n]))

            # compute positive part
            pos = model.jTmjp(
                positives,
                None,
                None,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            # compute negative part
            neg = model.jTmjp(
                negatives,
                None,
                None,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            # sum along batch size
            neg = torch.sum(neg, dim=0)

            return pos - neg


class ArccosHessianCalculator(HessianCalculator):
    """
    Contrastive Loss with normalization layer included, aka. arccos loss

    .. math::
        L(x,y) = 0.5 * sum_i x_i * y_i
               = 0.5 * || x / ||x|| - y / ||y|| || - 1

    .. math::
        Arcos(x, tuples) = \sum_{positives} L(x,y) - \sum_{negatives} L(x,y)

    .. note::
        arccos distance is equivalent to contrastive loss if the embeddings live on the l2-sphere,
        e.g. the last layer of the network is L2-normalization layer

    .. note::
        The arccos loss value is the same for method == "full" and method == "fix", however,
        the second order derivatives vary
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.method in ("full", "fix", "pos")

    @torch.no_grad()
    def compute_loss(
        self,
        x: torch.Tensor,
        model: nnj.Sequential,
        tuple_indices: Tuple,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute arccos loss

        .. math::
            L(x,y) = 0.5 * sum_i x_i * y_i
                = 0.5 * || x / ||x|| - y / ||y|| || - 1

        .. math::
            Arcos(x, tuples) = \sum_{positives} L(x,y) - \sum_{negatives} L(x,y)

        .. note::
            arccos distance is equivalent to contrastive loss if the embeddings live on the l2-sphere,
            e.g. the last layer of the network is L2-normalization layer

        Args:
            x: Images of the anchor and positive/negative samples.
            target: Embeddings of the anchor and positive/negative samples.
            model: Neural network module.
            tuple_indices: Tuple indices, either (a,p,n) or (a,p,a,n).
        """

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        # compute positive part
        pos = _arccos(model(x[ap]), model(x[p]))

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.method == "pos":
            return pos

        # compute negative part
        neg = _arccos(model(x[an]), model(x[n]))

        # sum along batch size
        neg = torch.sum(neg, dim=0)

        return pos - neg

    @torch.no_grad()
    def compute_gradient(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
        tuple_indices: Tuple,
    ) -> torch.Tensor:
        """
        Compute the gradient of the loss

        Args:
            x: Images of the anchor and positive/negative samples.
            target: Embeddings of the anchor and positive/negative samples.
            model: Neural network module.
            tuple_indices: Tuple indices, either (a,p,n) or (a,p,a,n).
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_hessian(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
        tuple_indices: Tuple,
    ) -> torch.Tensor:
        """
        Compute the hessian

        Args:
            x: Images of the anchor and positive/negative samples.
            target: Embeddings of the anchor and positive/negative samples.
            model: Neural network module.
            tuple_indices: Tuple indices, either (a,p,n) or (a,p,a,n).
        """
        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        if self.method == "full" or self.method == "pos":
            ###
            # compute positive part
            ###

            # forward pass
            z1, z2 = model(x[ap]), model(x[p])

            # initialize the hessian of the loss
            H = _arccos_hessian(z1, z2)

            # backpropagate through the network
            pos = model.jTmjp_batch2(
                x[ap],
                x[p],
                z1,
                z2,
                H,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                from_diag=False,
                diag_backprop=self.approximation_accuracy == "approx",
            )
            if self.hessian_shape == "diag":
                pos = pos[0] - 2 * pos[1] + pos[2]
            else:
                raise NotImplementedError
            # sum along batch size
            pos = torch.sum(pos, dim=0)

            if self.method == "pos":
                return pos

            ###
            # compute negative part
            ###

            # forward pass
            z1, z2 = model(x[an]), model(x[n])

            # initialize the hessian of the loss
            H = _arccos_hessian(z1, z2)

            # backpropagate through the network
            neg = model.jTmjp_batch2(
                x[an],
                x[n],
                z1,
                z2,
                H,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                from_diag=False,
                diag_backprop=self.approximation_accuracy == "approx",
            )
            if self.hessian_shape == "diag":
                neg = neg[0] - 2 * neg[1] + neg[2]
            else:
                raise NotImplementedError
            # sum along batch size
            neg = torch.sum(neg, dim=0)

            return pos - neg

        if self.method == "fix":
            ### compute positive part ###

            # forward pass
            z1, z2 = model(x[ap]), model(x[p])

            # initialize the hessian of the loss
            H1, _, H2 = _arccos_hessian(z1, z2)

            # backpropagate through the network
            pos1 = model.jTmjp(
                x[ap],
                None,
                H1,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            pos2 = model.jTmjp(
                x[p],
                None,
                H2,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            pos = pos1 + pos2

            # sum along batch size
            pos = torch.sum(pos, dim=0)

            ### compute negative part ###
            # forward pass
            z1, z2 = model(x[an]), model(x[n])

            # initialize the hessian of the loss
            H1, _, H2 = _arccos_hessian(z1, z2)

            # backpropagate through the network
            neg1 = model.jTmjp(
                x[an],
                None,
                H1,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            neg2 = model.jTmjp(
                x[n],
                None,
                H2,
                wrt="weight",
                to_diag=self.hessian_shape == "diag",
                diag_backprop=self.approximation_accuracy == "approx",
            )
            neg = neg1 + neg2

            # sum along batch size
            neg = torch.sum(neg, dim=0)

            return pos - neg
