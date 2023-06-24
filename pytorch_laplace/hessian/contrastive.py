import torch

from pytorch_laplace.hessian.base import HessianCalculator


def _arccos(z1, z2):
    z1_norm = torch.sum(z1**2, dim=1) ** (0.5)
    z2_norm = torch.sum(z2**2, dim=1) ** (0.5)
    return 0.5 * torch.einsum("bi,bi->b", z1, z2) / (z1_norm * z2_norm)


def _arccos_hessian(z1, z2):
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
    return tuple((H_11_normalized, H_12_normalized, H_22_normalized))


class ContrastiveHessianCalculator(HessianCalculator):
    """
    Contrastive Loss

    L(x,y) = 0.5 * || x - y ||
    Contrastive(x, tuples) = sum_positives L(x,y) - sum_negatives L(x,y)

    Notice that the contrastive loss value is the same for
        self.method == "full"
    and
        self.method == "fix"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.method in ("full", "fix", "pos")

    def compute_loss(self, x, target, nnj_module, tuple_indices):
        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        # compute positive part
        pos = 0.5 * (nnj_module(x[ap]) - nnj_module(x[p])) ** 2

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.method == "pos":
            return pos

        # compute negative part
        neg = 0.5 * (nnj_module(x[an]) - nnj_module(x[n])) ** 2

        # sum along batch size
        neg = torch.sum(neg, dim=0)

        return pos - neg

    def compute_gradient(self, x, target, nnj_module, tuple_indices):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices):
        with torch.no_grad():
            # unpack tuple indices
            if len(tuple_indices) == 3:
                a, p, n = tuple_indices
                ap = an = a
            else:
                ap, p, an, n = tuple_indices
            assert len(ap) == len(p) and len(an) == len(n)

            if self.method == "full" or self.method == "pos":
                # compute positive part
                pos = nnj_module._jTmjp_batch2(
                    x[ap],
                    x[p],
                    None,
                    None,
                    None,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                if self.shape == "diagonal":
                    pos = pos[0] - 2 * pos[1] + pos[2]
                else:
                    raise NotImplementedError
                # sum along batch size
                pos = torch.sum(pos, dim=0)

                if self.method == "pos":
                    return pos

                # compute negative part
                neg = nnj_module._jTmjp_batch2(
                    x[an],
                    x[n],
                    None,
                    None,
                    None,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                if self.shape == "diagonal":
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
                pos = nnj_module._jTmjp(
                    positives,
                    None,
                    None,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                # sum along batch size
                pos = torch.sum(pos, dim=0)

                # compute negative part
                neg = nnj_module._jTmjp(
                    negatives,
                    None,
                    None,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                # sum along batch size
                neg = torch.sum(neg, dim=0)

                return pos - neg


class ArccosHessianCalculator(HessianCalculator):
    """
    Contrastive Loss with normalization layer included, aka. arccos loss
    L(x,y) = 0.5 * sum_i x_i * y_i
            = 0.5 * || x / ||x|| - y / ||y|| || - 1    # arccos distance is equivalent to contrastive distance & normalization layer
    Arcos(x, tuples) = sum_positives L(x,y) - sum_negatives L(x,y)

    Notice that the arccos loss value is the same for
        self.method == "full"
    and
        self.method == "fix"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.method in ("full", "fix", "pos")

    def compute_loss(self, x, nnj_module, tuple_indices):
        """
        L(x,y) = 0.5 * sum_i x_i * y_i
               = 0.5 * || x / ||x|| - y / ||y|| || - 1    # arccos distance is equivalent to contrastive distance & normalization layer
        Arcos(x, tuples) = sum_positives L(x,y) - sum_negatives L(x,y)
        """

        # unpack tuple indices
        if len(tuple_indices) == 3:
            a, p, n = tuple_indices
            ap = an = a
        else:
            ap, p, an, n = tuple_indices
        assert len(ap) == len(p) and len(an) == len(n)

        # compute positive part
        pos = _arccos(nnj_module(x[ap]), nnj_module(x[p]))

        # sum along batch size
        pos = torch.sum(pos, dim=0)

        if self.method == "pos":
            return pos

        # compute negative part
        neg = _arccos(nnj_module(x[an]), nnj_module(x[n]))

        # sum along batch size
        neg = torch.sum(neg, dim=0)

        return pos - neg

    def compute_gradient(self, x, target, nnj_module, tuple_indices):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices):
        with torch.no_grad():
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
                z1, z2 = nnj_module(x[ap]), nnj_module(x[p])

                # initialize the hessian of the loss
                H = _arccos_hessian(z1, z2)

                # backpropagate through the network
                pos = nnj_module._jTmjp_batch2(
                    x[ap],
                    x[p],
                    z1,
                    z2,
                    H,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    from_diag=False,
                    diag_backprop=self.speed == "fast",
                )
                if self.shape == "diagonal":
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
                z1, z2 = nnj_module(x[an]), nnj_module(x[n])

                # initialize the hessian of the loss
                H = _arccos_hessian(z1, z2)

                # backpropagate through the network
                neg = nnj_module._jTmjp_batch2(
                    x[an],
                    x[n],
                    z1,
                    z2,
                    H,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    from_diag=False,
                    diag_backprop=self.speed == "fast",
                )
                if self.shape == "diagonal":
                    neg = neg[0] - 2 * neg[1] + neg[2]
                else:
                    raise NotImplementedError
                # sum along batch size
                neg = torch.sum(neg, dim=0)

                return pos - neg

            if self.method == "fix":
                ### compute positive part ###

                # forward pass
                z1, z2 = nnj_module(x[ap]), nnj_module(x[p])

                # initialize the hessian of the loss
                H1, _, H2 = _arccos_hessian(z1, z2)

                # backpropagate through the network
                pos1 = nnj_module._jTmjp(
                    x[ap],
                    None,
                    H1,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                pos2 = nnj_module._jTmjp(
                    x[p],
                    None,
                    H2,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                pos = pos1 + pos2

                # sum along batch size
                pos = torch.sum(pos, dim=0)

                ### compute negative part ###
                # forward pass
                z1, z2 = nnj_module(x[an]), nnj_module(x[n])

                # initialize the hessian of the loss
                H1, _, H2 = _arccos_hessian(z1, z2)

                # backpropagate through the network
                neg1 = nnj_module._jTmjp(
                    x[an],
                    None,
                    H1,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                neg2 = nnj_module._jTmjp(
                    x[n],
                    None,
                    H2,
                    wrt=self.wrt,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                neg = neg1 + neg2

                # sum along batch size
                neg = torch.sum(neg, dim=0)

                return pos - neg
