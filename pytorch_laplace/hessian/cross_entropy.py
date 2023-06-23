import torch
from pytorch_laplace.hessian.base import HessianCalculator


class CEHessianCalculator(HessianCalculator):
    " Multi-Class Cross Entropy "
    # only support one point prediction (for now)
    # for example: 
    #       - mnist classification: OK
    #       - image pixelwise classification: NOT OK
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.method == ""

    def compute_loss(self, x, target, nnj_module, tuple_indices=None, reshape=None):

        with torch.no_grad():
            val = nnj_module(x)
            if reshape is not None:
                val = val.reshape(val.shape[0], *reshape)
            assert val.shape == target.shape
            if len(val.shape)!=2 and len(val.shape)!=3:
                raise ValueError("Ei I need logits to be either 1d or 2d tensors (+ batch size)")

            log_normalization = torch.log(torch.sum(torch.exp(val), dim = -1)).unsqueeze(-1).expand(val.shape)
            cross_entropy = -(target * val) + log_normalization
            #print(torch.sum(log_normalization), torch.sum(target * val))
            cross_entropy = torch.sum(cross_entropy, dim=-1)

            # average along multiple points (if any)
            if len(val.shape)==3:
                cross_entropy = torch.mean(cross_entropy, dim=1)
            # average along batch size
            cross_entropy = torch.mean(cross_entropy, dim=0)
            return cross_entropy

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None, reshape=None):
        
        with torch.no_grad():
            val = nnj_module(x)
            if reshape is not None:
                val = val.reshape(val.shape[0], *reshape)
            assert val.shape == target.shape

            exp_val = torch.exp(val)
            softmax = torch.einsum("b...i,b...->b...i", exp_val, 1./torch.sum(exp_val, dim = -1) )

            # compute gradient of the Bernoulli log-likelihood
            gradient = softmax - target

            # backpropagate through the network
            gradient = gradient.reshape(val.shape[0], -1)
            gradient = nnj_module._vjp(x, val, gradient, wrt=self.wrt)

            # average along batch size
            gradient = torch.mean(gradient, dim=0)
            return gradient

    def compute_hessian(self, x, nnj_module, tuple_indices=None, save_memory=False, reshape=None):

        with torch.no_grad():
            val = nnj_module(x)
            if reshape is not None:
                val = val.reshape(val.shape[0], *reshape)

            if len(val.shape)==2:
                #single point classification

                exp_val = torch.exp(val)
                softmax = torch.einsum("bi,b->bi", exp_val, 1./torch.sum(exp_val, dim = 1) )

                # hessian = diag(softmax) - softmax.T * softmax
                # thus Jt * hessian * J = Jt * diag(softmax) * J - Jt * softmax.T * softmax * J


                # backpropagate through the network the diagonal part
                Jt_diag_J = nnj_module._jTmjp(
                    x,
                    val,
                    softmax,
                    wrt=self.wrt,
                    from_diag=True,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                # backpropagate through the network the outer product   
                softmax_J = nnj_module._vjp(
                    x,
                    val,
                    softmax,
                    wrt=self.wrt
                )
                if self.shape == "diagonal":
                    Jt_outer_J = torch.einsum("bi,bi->bi", softmax_J, softmax_J)
                else:
                    Jt_outer_J = torch.einsum("bi,bj->bij", softmax_J, softmax_J)

                # add the backpropagated quantities
                Jt_H_J = Jt_diag_J - Jt_outer_J

                # average along batch size
                Jt_H_J = torch.mean(Jt_H_J, dim=0)
                return Jt_H_J

            if len(val.shape)==3:
                #multi point classification

                b, p, c = val.shape

                exp_val = torch.exp(val)
                softmax = torch.einsum("bpi,bp->bpi", exp_val, 1./torch.sum(exp_val, dim = 2) )

                # backpropagate through the network the diagonal part
                diagonal = softmax.reshape(val.shape[0], val.shape[1:].numel())
                Jt_diag_J = nnj_module._jTmjp(
                    x,
                    val,
                    diagonal,
                    wrt=self.wrt,
                    from_diag=True,
                    to_diag=self.shape == "diagonal",
                    diag_backprop=self.speed == "fast",
                )
                # backpropagate through the network the outer product  
                if save_memory is True:
                    Jt_outer_J = torch.zeros_like(Jt_diag_J)
                    for point in range(p):
                        vector = torch.zeros(b,p*c, device=val.device)
                        vector[:, point*c : (point+1)*c] = softmax[:, point, :]
                        softmax_J = nnj_module._vjp(
                            x,
                            val,
                            vector,
                            wrt=self.wrt
                        )
                        if self.shape == "diagonal":
                            Jt_outer_J += torch.einsum("bi,bi->bi", softmax_J, softmax_J)
                        else:
                            Jt_outer_J += torch.einsum("bi,bj->bij", softmax_J, softmax_J)
                elif save_memory is False:
                    pos_identity = torch.diag_embed(torch.ones(p, device=val.device))
                    matrix = torch.einsum("bpi,pq->bpqi", softmax, pos_identity).reshape(b, p, p*c)
                    softmax_J = nnj_module._mjp(
                        x,
                        val,
                        matrix,
                        wrt=self.wrt
                    )
                    if self.shape == "diagonal":
                        Jt_outer_J = torch.einsum("bki,bki->bi", softmax_J, softmax_J)
                    else:
                        Jt_outer_J = torch.einsum("bki,bkj->bij", softmax_J, softmax_J)
                else:
                    Jt_outer_J = torch.zeros_like(Jt_diag_J)
                    batch_size = int(p/save_memory)
                    assert batch_size == p/save_memory
                    pos_identity = torch.diag_embed(torch.ones(batch_size, device=val.device))
                    for batch_n in range(save_memory):
                        matrix = torch.einsum("bpi,pq->bpqi", 
                                            softmax[:,batch_n*batch_size:(batch_n+1)*batch_size,:], 
                                            pos_identity).reshape(b, batch_size, batch_size*c)
                        matrix = torch.cat([torch.zeros(b, batch_size, (batch_n*batch_size)*c, device=val.device),
                                            matrix,
                                            torch.zeros(b, batch_size, ((save_memory-batch_n-1)*batch_size)*c, device=val.device)], dim=2)
                        softmax_J = nnj_module._mjp(
                            x,
                            val,
                            matrix,
                            wrt=self.wrt
                        )
                        if self.shape == "diagonal":
                            Jt_outer_J += torch.einsum("bki,bki->bi", softmax_J, softmax_J)
                        else:
                            Jt_outer_J += torch.einsum("bki,bkj->bij", softmax_J, softmax_J)

                # add the backpropagated quantities
                Jt_H_J = Jt_diag_J - Jt_outer_J

                # average along batch size
                Jt_H_J = torch.mean(Jt_H_J, dim=0)
                return Jt_H_J