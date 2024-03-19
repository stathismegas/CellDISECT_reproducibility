import random
from typing import Callable, Iterable, Literal, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence as kl
from torchmetrics import Accuracy, F1Score

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers


torch.backends.cudnn.benchmark = True
from .utils_m import *
from scvi.module._classifier import Classifier

dim_indices = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define DecoderPoissonVI module for atac

#from scvi.module._peakvae import Decoder as DecoderPeakVI
#from poisson_atac.module._poissonvae import DecoderPoissonVI



class Dis2pmPoissonVAE2(BaseModuleClass):
    """
    Variational auto-encoder module. 

    It differs from Dis2pmPoissonVAE by using the scviDecoders, not the poissonVAE decoders

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent_shared
        Dimensionality of the shared latent space (Z_{-s}) 
    n_latent_attribute
        Dimensionality of the latent space for each sensitive attributes (Z_{s_i})
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covariates
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    region_likelihood
        One of
        * ``'bernoulli'`` - Bernoulli distribution (implemented in PeakVAE and MultiVAE)
        * ``'poisson'`` - Poisson distribution (implemented in PoissonVAE) 
    latent_distribution
        One of
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
            self,
            n_input_genes: int,
            n_input_regions: int,
            n_hidden: Tunable[int] = 128,
            n_latent_shared: Tunable[int] = 10,
            n_latent_attribute: Tunable[int] = 10,
            n_layers: Tunable[int] = 1,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate: Tunable[float] = 0.1,
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.n_input_regions = n_input_regions
        self.n_input_genes = n_input_genes
        
        self.dispersion = "gene"
        self.n_latent_shared = n_latent_shared
        self.n_latent_attribute = n_latent_attribute
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.px_r = torch.nn.Parameter(torch.randn(n_input_genes)).to(device)
        self.px_r_acc = torch.nn.Parameter(torch.randn(n_input_regions)).to(device)

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoders for genes
        # Use Encoders with return_distr=True so that they return 
        # the whole distribution named with qz below
        # and a sample from it named z

        self.n_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        self.zs_num = len(self.n_cat_list)
        
        print(n_input_genes)
        print("debug4")
        print(f'n_cats_per_cov is {n_cats_per_cov}')

        # print("debug4")
        n_input_encoder_g = n_input_genes
        # print("debug5")

        self.z_encoders_list = nn.ModuleList(
            [
                Encoder(
                    n_input_encoder_g,
                    n_latent_shared,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
            ]
        )

        self.z_encoders_list.extend(
            [
                Encoder(
                    n_input_encoder_g,
                    n_latent_attribute,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.z_prior_encoders_list = nn.ModuleList(
            [
                Encoder(
                    0,
                    n_latent_attribute,
                    n_cat_list=[self.n_cat_list[k]],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )
        
        
        # Encoders for regions

        n_input_encoder_r = n_input_regions
        # print(n_input_encoder_r)

        self.z_encoders_list_acc = nn.ModuleList(
            [
                Encoder(
                    n_input_encoder_r,
                    n_latent_shared,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates, #MultiVI comments this out for atac
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
            ]
        )

        self.z_encoders_list_acc.extend(
            [
                Encoder(
                    n_input_encoder_r,
                    n_latent_attribute,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates, #MultiVI comments this out for atac
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.z_prior_encoders_list_acc = nn.ModuleList(
            [
                Encoder(
                    0,
                    n_latent_attribute,
                    n_cat_list=[self.n_cat_list[k]],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )
               
        

        # Decoders for genes

        self.x_decoders_list = nn.ModuleList(
            [
                DecoderSCVI(
                    n_latent_shared,
                    n_input_genes,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
            ]
        )

        self.x_decoders_list.extend(
            [
                DecoderSCVI(
                    n_latent_attribute,
                    n_input_genes,
                    n_cat_list=[self.n_cat_list[i] for i in range(len(self.n_cat_list)) if i != k],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.n_latent = n_latent_shared + n_latent_attribute * self.zs_num

        self.s_classifiers_list = nn.ModuleList([])
        for i in range(self.zs_num):
            self.s_classifiers_list.append(
                Classifier(
                    n_input=n_latent_attribute,
                    n_labels=self.n_cat_list[i],
                ).to(device)
            )

            
        # Decoders for regions
        
        self.x_decoders_list_acc = nn.ModuleList(
            [
                DecoderSCVI(
                    n_latent_shared,
                    n_input_regions,
                    n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
            ]
        )

        self.x_decoders_list_acc.extend(
            [
                DecoderSCVI(
                    n_latent_attribute,
                    n_input_regions,
                    n_cat_list=[self.n_cat_list[i] for i in range(len(self.n_cat_list)) if i != k],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.n_latent = n_latent_shared + n_latent_attribute * self.zs_num

        self.s_classifiers_list_acc = nn.ModuleList([])
        for i in range(self.zs_num):
            self.s_classifiers_list_acc.append(
                Classifier(
                    n_input=n_latent_attribute,
                    n_labels=self.n_cat_list[i],
                ).to(device)
            )



            
    def _get_inference_input(self, tensors):
        
        #print("inside _get_inference_input")

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key]

        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = {
            "x": x,
            "cat_covs": cat_covs,
        }
        return input_dict
    

    
    @auto_move_data
    def inference(self, x,
                  cat_covs,
                  nullify_cat_covs_indices: Optional[List[int]] = None,
                  nullify_shared: Optional[bool] = False,
                  ):
        
        if self.n_input_genes == 0:
            x_rna = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_rna = x[:, : self.n_input_genes]
        if self.n_input_regions == 0:
            x_chr = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_chr = x[
                :, self.n_input_genes : (self.n_input_genes + self.n_input_regions)
            ]
            

        # # pass covariates to encoders? 
        # if cont_covs is not None and self.encode_covariates:
        #     encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
        #     encoder_input_accessibility = torch.cat((x_chr, cont_covs), dim=-1)
        #     encoder_input_protein = torch.cat((y, cont_covs), dim=-1)
        # else:
        #     encoder_input_expression = x_rna
        #     encoder_input_accessibility = x_chr
        #     encoder_input_protein = y
    
        nullify_cat_covs_indices = [] if nullify_cat_covs_indices is None else nullify_cat_covs_indices

        x_ = x_rna
        x_r = x_chr
        # print(f'x_r is {x_r}')
        
        # log-transform the RNA
        library = torch.log(x_.sum(1)).unsqueeze(1)
        #print(f'{library.size()}'+' is the size of library' )
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # define batch size
        batch_size = x.size(dim=0)
       
        # log-transform the atac
        library_acc = torch.log(x_r.sum(1)).unsqueeze(1).to(device) #torch.unsqueeze(x.sum(1),1) 
        if self.log_variational:
            x_r = torch.log(1 + x_r)

        
        cat_in = torch.split(cat_covs, 1, dim=1)

        # z_shared, z_shared_acc: qz is the distribution, z is a sample from it

        qz_shared, z_shared = self.z_encoders_list[0](x_, *cat_in)
        z_shared = z_shared.to(device)

        qz_shared_acc, z_shared_acc = self.z_encoders_list_acc[0](x_r, *cat_in)

        z_shared_acc = z_shared_acc.to(device)

        # zs

        encoders_outputs = []
        encoders_inputs = [(x_, *cat_in) for _ in cat_in]

        
        for i in range(len(self.z_encoders_list) - 1):
            encoders_outputs.append(self.z_encoders_list[i + 1](*encoders_inputs[i]))

        qzs = [enc_out[0] for enc_out in encoders_outputs]
        zs = [enc_out[1].to(device) for enc_out in encoders_outputs]

        # zs_acc
        
        encoders_outputs_acc = []
        encoders_inputs_acc = [(x_r, *cat_in) for _ in cat_in]

        for i in range(len(self.z_encoders_list_acc) - 1):
            encoders_outputs_acc.append(self.z_encoders_list_acc[i + 1](*encoders_inputs_acc[i]))

        qzs_acc = [enc_out[0] for enc_out in encoders_outputs_acc]
        zs_acc = [enc_out[1].to(device) for enc_out in encoders_outputs_acc]
        
        # zs_prior

        encoders_prior_outputs = []
        encoders_prior_inputs = [(torch.tensor([]).to(device), c) for c in cat_in]
        for i in range(len(self.z_prior_encoders_list)):
            encoders_prior_outputs.append(self.z_prior_encoders_list[i](*encoders_prior_inputs[i]))

        qzs_prior = [enc_out[0] for enc_out in encoders_prior_outputs]
        zs_prior = [enc_out[1].to(device) for enc_out in encoders_prior_outputs]

        
        # zs_prior_acc

        encoders_prior_outputs_acc = []
        encoders_prior_inputs_acc = [(torch.tensor([]).to(device), c) for c in cat_in]
        for i in range(len(self.z_prior_encoders_list_acc)):
            encoders_prior_outputs_acc.append(self.z_prior_encoders_list_acc[i](*encoders_prior_inputs_acc[i]))

        qzs_prior_acc = [enc_out[0] for enc_out in encoders_prior_outputs_acc]
        zs_prior_acc = [enc_out[1].to(device) for enc_out in encoders_prior_outputs_acc]

        
                
        # nullify if required

        if nullify_shared:
            z_shared = torch.zeros_like(z_shared).to(device)
            z_shared_acc = torch.zeros_like(z_shared_acc).to(device)

        for i in range(self.zs_num):
            if i in nullify_cat_covs_indices:
                zs[i] = torch.zeros_like(zs[i]).to(device)
                zs_acc[i] = torch.zeros_like(zs_acc[i]).to(device)

        zs_concat = torch.cat(zs, dim=-1)
        z_concat = torch.cat([z_shared, zs_concat], dim=-1)

        zs_concat_acc = torch.cat(zs_acc, dim=-1)
        z_concat_acc = torch.cat([z_shared_acc, zs_concat_acc], dim=-1)
        
        
        output_dict = {
            "z_shared": z_shared,
            "zs": zs,
            "zs_prior": zs_prior,
            "qz_shared": qz_shared,
            "qzs": qzs,
            "qzs_prior": qzs_prior,
            "z_concat": z_concat,
            "library": library,
            
           "z_shared_acc": z_shared_acc,
            "zs_acc": zs_acc,
            "zs_prior_acc": zs_prior_acc,
            "qz_shared_acc": qz_shared_acc,
            "qzs_acc": qzs_acc,
            "qzs_prior_acc": qzs_prior_acc,
            "z_concat_acc": z_concat_acc,
            "library_acc": library_acc,
            
            "cat_covs": cat_covs,
        }
        return output_dict


    def _get_generative_input(self, tensors, inference_outputs):
        #print("inside _get_generative_input")
        input_dict = {
            "z_shared": inference_outputs["z_shared"],
            "zs": inference_outputs["zs"],  # a list of all zs
            "library": inference_outputs["library"],
            
            "cat_covs": inference_outputs["cat_covs"],
            
            "z_shared_acc": inference_outputs["z_shared_acc"],
            "zs_acc": inference_outputs["zs_acc"],  # a list of all zs
            "library_acc": inference_outputs["library_acc"],
             
        }
        return input_dict


    
    @auto_move_data
    def generative(self, 
                   z_shared, zs, library,
                   cat_covs,
                   z_shared_acc, zs_acc, library_acc,
                   ):

        output_dict = {"px": [],  
                       "px_acc": [] }
        

        z = [z_shared] + zs
        z_acc = [z_shared_acc] + zs_acc


        cats_splits = torch.split(cat_covs, 1, dim=1) # returns tuple of tensors where each tensor has the values of a specific covariate
        all_cats_but_one = []
        for i in range(self.zs_num):
            all_cats_but_one.append([cats_splits[j] for j in range(len(cats_splits)) if j != i])

        dec_cats_in = [cats_splits] + all_cats_but_one

        for dec_count in range( self.zs_num + 1):  ### just for debugging set to range(1, self.zs_num + 1)
            
            dec_covs = dec_cats_in[dec_count]
            
            # ----------------------- For gene expression -----------------------

            x_decoder = self.x_decoders_list[dec_count]
            x_decoder_input = z[dec_count]
            #print(f'x_decoder_input is : {x_decoder_input.size()}')
            #print(f'dec_covs is : {len(dec_covs)}')
            #print(f'dec_covs is : {dec_covs}')

            px_scale, px_r, px_rate, px_dropout = x_decoder(
                self.dispersion,
                x_decoder_input,
                library,
                *dec_covs
            )
            px_r = torch.exp(self.px_r)

            if self.gene_likelihood == "zinb":
                px = ZeroInflatedNegativeBinomial(
                    mu=px_rate,
                    theta=px_r,
                    zi_logits=px_dropout,
                    scale=px_scale,
                )
            elif self.gene_likelihood == "nb":
                px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
            elif self.gene_likelihood == "poisson":
                px = Poisson(px_rate, scale=px_scale)

            output_dict["px"] += [px]
            
            
            # ----------------------- For accessibility -----------------------
            

            # #print(f'x_decoder_acc is : {x_decoder_acc}')
            # #print(f'x_decoder is : {x_decoder}')       
            
            # #print(f'x_decoder_input_acc is : {x_decoder_input_acc.size()}')
            # #print(f'the size of x_decoder_input_acc is : {x_decoder_input_acc}')
            # #print(f'dec_covs is : {dec_covs}')
            # #print(f'library_acc is : {library_acc}')
            # #print(f'size of library_acc is {library_acc.size()}')



            # y_scale, _, px_acc, _ = x_decoder_acc(
            #     x_decoder_input_acc, library_acc, *dec_covs
            # )        
                
            
            # # px_acc = x_decoder_acc(
            # #     x_decoder_input_acc,
            # #     *dec_covs
            # # )

            
            # #output_dict["y_scale"] += [y_scale]  
            # output_dict["px_acc"] += [px_acc] 
            # # ---------------

            x_decoder_acc = self.x_decoders_list_acc[dec_count]            
            x_decoder_input_acc = z_acc[dec_count]
            
            px_scale_acc, px_r_acc, px_rate_acc, px_dropout_acc = x_decoder_acc(
                self.dispersion,
                x_decoder_input_acc,
                library_acc,
                *dec_covs
            )
            px_r_acc = torch.exp(self.px_r_acc)

            if self.gene_likelihood == "zinb":
                px_acc = ZeroInflatedNegativeBinomial(
                    mu=px_rate_acc,
                    theta=px_r_acc,
                    zi_logits=px_dropout_acc,
                    scale=px_scale_acc,
                )
            elif self.gene_likelihood == "nb":
                px = NegativeBinomial(mu=px_rate_acc, theta=px_r_acc, scale=px_scale_acc)
            elif self.gene_likelihood == "poisson":
                px = Poisson(px_rate_acc, scale=px_scale_acc)

            output_dict["px_acc"] += [px_acc]
        



        return output_dict

    def sub_forward(self, idx,
                    x, cat_covs,
                    detach_x=False,
                    detach_z=False):
        """

        performs forward (inference + generative) only on enc/dec idx for gene expression

        Parameters
        ----------
        idx
            index of enc/dec in [1, ..., self.zs_num]
        x
        cat_covs
        detach_x
        detach_z

        """
        #print("inside sub_forward")
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        cat_in = torch.split(cat_covs, 1, dim=1)

        qz, z = (self.z_encoders_list[idx](x_, *cat_in))
        if detach_z:
            z = z.detach()

        dec_cats = [cat_in[j] for j in range(len(cat_in)) if j != idx-1]

        x_decoder = self.x_decoders_list[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            z,
            library,
            *dec_cats
        )
        px_r = torch.exp(self.px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
            
        return px




    def sub_forward_acc(self, idx,
                    x, cat_covs,
                    detach_x=False,
                    detach_z=False):
        """

        performs forward (inference + generative) only on enc/dec idx for atac

        Parameters
        ----------
        idx
            index of enc/dec in [1, ..., self.zs_num]
        x
        cat_covs
        detach_x
        detach_z

        """
        #print("inside sub_forward")
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        cat_in = torch.split(cat_covs, 1, dim=1)

        qz, z = (self.z_encoders_list_acc[idx](x_, *cat_in))
        if detach_z:
            z = z.detach()

        dec_cats = [cat_in[j] for j in range(len(cat_in)) if j != idx-1]

        x_decoder = self.x_decoders_list_acc[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            z,
            library,
            *dec_cats
        )
        px_r = torch.exp(self.px_r_acc)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
            
        return px

    
    # def sub_forward_acc(self, idx,
    #                 x, cat_covs,
    #                 detach_x=False,
    #                 detach_z=False):
    #     """

    #     performs forward (inference + generative) only on enc/dec idx for accessibility

    #     Parameters
    #     ----------
    #     idx
    #         index of enc/dec in [1, ..., self.zs_num]
    #     x
    #     cat_covs
    #     detach_x
    #     detach_z

    #     """
        
    #     x_ = x
    #     if detach_x:
    #         x_ = x.detach()
    #     # print("detached x")

    #     library_acc = torch.log(x_.sum(1)).unsqueeze(1).to(device) 

    #     cat_in = torch.split(cat_covs, 1, dim=1)
    #     # print("torch.split ")

    #     qz, z = (self.z_encoders_list_acc[idx](x_, *cat_in))
    #     if detach_z:
    #         z = z.detach()
    #     # print("detached z")

    #     dec_cats = [cat_in[j] for j in range(len(cat_in)) if j != idx-1]
    #     # print("dec_cats ")

    #     x_decoder_acc = self.x_decoders_list_acc[idx]
    #     # print("x_decoder ")     

    #     y_scale, _, px_acc, _ = x_decoder_acc(
    #         z, library_acc, *dec_cats
    #     )        
                
    #     return px_acc
    
    

    def classification_logits(self, inference_outputs):
        zs = inference_outputs["zs"]
        logits = []
        for i in range(self.zs_num):
            s_i_classifier = self.s_classifiers_list[i]
            logits_i = s_i_classifier(zs[i])
            logits += [logits_i]

        return logits

    def classification_logits_acc(self, inference_outputs):
        zs = inference_outputs["zs_acc"]
        logits = []
        for i in range(self.zs_num):
            s_i_classifier = self.s_classifiers_list_acc[i]
            logits_i = s_i_classifier(zs[i])
            logits += [logits_i]

        return logits
    
    def compute_clf_metrics(self, logits, cat_covs):
        # CE, ACC, F1
        cats = torch.split(cat_covs, 1, dim=1)
        ce_losses = []
        accuracy_scores = []
        f1_scores = []
        for i in range(self.zs_num):
            s_i = one_hot_cat([self.n_cat_list[i]], cats[i]).to(device)
            ce_losses += [F.cross_entropy(logits[i], s_i)]
            kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
            predicted_labels = torch.argmax(logits[i], dim=-1, keepdim=True).to(device)
            acc = Accuracy(**kwargs).to(device)
            accuracy_scores.append(acc(predicted_labels, cats[i]).to(device))
            F1 = F1Score(**kwargs).to(device)
            f1_scores.append(F1(predicted_labels, cats[i]).to(device))

        ce_loss_sum = sum(torch.mean(ce) for ce in ce_losses)
        accuracy = sum(accuracy_scores) / len(accuracy_scores)
        f1 = sum(f1_scores) / len(f1_scores)

        return ce_loss_sum, accuracy, f1
    
    # def get_reconstruction_loss_accessibility(self, x, px_acc):
    #     """Computes the reconstruction loss for the accessibility data."""
    #     #print("inside get_reconstruction_loss_accessibility")

    #     reconst_loss = Poisson(px_acc).log_prob(x).sum(dim=-1)

    #     return reconst_loss
        
        #reg_factor = (
        #    torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        #)
        
#         print(f'the p is {p}')
#         print(f'the d is {d}')
#         print(f'the x is {x}')
#         print(f'the x size is {x.size()}')
#         d_unlist = [item[0] for item in d]
#         print(f'the d size is {d.size()}')
#         print(f'the p[1] size is {p[1].size()}')
#         print(f'the len(p) is {len(p)}')

#         print(f'the d_unlist is {d_unlist}')
        # print(f'the p size is {p.size()}')
        # print(f'the x size is {x.size()}')

        # return torch.nn.BCELoss(reduction="none")(
        #     #p * d , (x > 0).float()
        #     p  , (x > 0).float()
        # ).sum(dim=-1)
    
    

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            cf_weight: Tunable[Union[float, int]],  # RECONST_LOSS_X_CF weight
            beta: Tunable[Union[float, int]],  # KL Zi weight
            clf_weight: Tunable[Union[float, int]],  # Si classifier weight
            n_cf: Tunable[int],  # number of X_cf recons (X_cf = a random permutation of X)
            kl_weight: float = 1.0,
    ):
        #print("inside loss")

        x = tensors[REGISTRY_KEYS.X_KEY]
        
        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        
        # print(f'x_chr is {x_chr}')

        #print(generative_outputs["px"])
        #print("\n")
        #print("now the accessibility")
        #print(generative_outputs["px_acc"])
        #print("\n")
        #print(generative_outputs)

        # ATAC Loss --------------------------------------------
        
        # ATAC Reconstruction loss
        reconst_loss_x_list_acc = [-torch.mean(px.log_prob(x_chr).sum(-1)) for px in generative_outputs["px_acc"]]
        reconst_loss_x_dict_acc = {'atac_' + str(i): reconst_loss_x_list_acc[i] for i in range(len(reconst_loss_x_list_acc))}
        reconst_loss_x_acc = sum(reconst_loss_x_list_acc)
    
    
        # ATAC reconstruction loss X'
        cat_covs = tensors[REGISTRY_KEYS.CAT_COVS_KEY]
        batch_size = x.size(dim=0)

        reconst_loss_x_cf_list_acc = []

        for _ in range(n_cf):

            # choose a random permutation of X as X_cf

            idx_shuffled = list(range(batch_size))
            random.shuffle(idx_shuffled)
            idx_shuffled = torch.tensor(idx_shuffled).to(device)

            x_ = x_chr
            x_cf_acc = torch.index_select(x_chr, 0, idx_shuffled).to(device)

            cat_cov_ = cat_covs
            cat_cov_cf = torch.index_select(cat_covs, 0, idx_shuffled).to(device)
            cat_cov_cf_split = torch.split(cat_cov_cf, 1, dim=1)

            # a random ordering for diffusing through n VAEs

            perm = list(range(self.zs_num))
            random.shuffle(perm)

            for idx in perm:
                # cat_cov_[idx] (possibly) changes to cat_cov_cf[idx]
                cat_cov_split = list(torch.split(cat_cov_, 1, dim=1))
                cat_cov_split[idx] = cat_cov_cf_split[idx]
                cat_cov_ = torch.cat(cat_cov_split, dim=1)
                # use enc/dec idx+1 to get px_ and feed px_.mean as the next x_
                px_acc_ = self.sub_forward_acc(idx + 1, x_, cat_cov_)
                x_acc_ = px_acc_.mean

            reconst_loss_x_cf_list_acc.append(-torch.mean(px_acc_.log_prob(x_cf_acc).sum(-1)))

        # print(f'reconst_loss_x_cf_list_acc before sum is {reconst_loss_x_cf_list_acc}')
        reconst_loss_x_cf_acc = sum(reconst_loss_x_cf_list_acc) / n_cf
        
        
        # ATAC KL divergence Z
        # print("ATAC KL divergence Z ")

        kl_z_list_acc = [torch.mean(kl(qzs_acc, qzs_prior_acc).sum(dim=1)) for qzs_acc, qzs_prior_acc in
                     zip(inference_outputs["qzs_acc"], inference_outputs["qzs_prior_acc"])]

        kl_z_dict_acc = {'z_acc_' + str(i+1): kl_z_list_acc[i] for i in range(len(kl_z_list_acc))}

        # ATAC classification metrics: CE, ACC, F1

        logits_acc = self.classification_logits_acc(inference_outputs)
        ce_loss_sum_acc, accuracy_acc, f1_acc = self.compute_clf_metrics(logits_acc, cat_covs)
        ce_loss_mean_acc = ce_loss_sum_acc / len(range(self.zs_num))        
          
        
        
        # Gene expression Loss ---------------------------------
        
        # reconstruction loss X
        # print("GEX reconstruction loss X ")

        reconst_loss_x_list = [-torch.mean(px.log_prob(x_rna).sum(-1)) for px in generative_outputs["px"]]
        reconst_loss_x_dict = {'x_' + str(i): reconst_loss_x_list[i] for i in range(len(reconst_loss_x_list))}
        reconst_loss_x = sum(reconst_loss_x_list)

        # reconstruction loss X'
        # print("GEX reconstruction loss X cf ")

        # cat_covs = tensors[REGISTRY_KEYS.CAT_COVS_KEY]
        # batch_size = x.size(dim=0)

        reconst_loss_x_cf_list = []

        for _ in range(n_cf):

            # choose a random permutation of X as X_cf

            idx_shuffled = list(range(batch_size))
            random.shuffle(idx_shuffled)
            idx_shuffled = torch.tensor(idx_shuffled).to(device)

            x_ = x_rna
            x_cf = torch.index_select(x_rna, 0, idx_shuffled).to(device)

            cat_cov_ = cat_covs
            cat_cov_cf = torch.index_select(cat_covs, 0, idx_shuffled).to(device)
            cat_cov_cf_split = torch.split(cat_cov_cf, 1, dim=1)

            # a random ordering for diffusing through n VAEs

            perm = list(range(self.zs_num))
            random.shuffle(perm)

            for idx in perm:
                # cat_cov_[idx] (possibly) changes to cat_cov_cf[idx]
                cat_cov_split = list(torch.split(cat_cov_, 1, dim=1))
                cat_cov_split[idx] = cat_cov_cf_split[idx]
                cat_cov_ = torch.cat(cat_cov_split, dim=1)
                # use enc/dec idx+1 to get px_ and feed px_.mean as the next x_
                px_ = self.sub_forward(idx + 1, x_, cat_cov_)
                x_ = px_.mean

            reconst_loss_x_cf_list.append(-torch.mean(px_.log_prob(x_cf).sum(-1)))
            
        # print(f'reconst_loss_x_cf_list before sum is {reconst_loss_x_cf_list}')
        reconst_loss_x_cf = sum(reconst_loss_x_cf_list) / n_cf

        # KL divergence Z
        # print("GEX reconstruction loss KL divergence Z ")

        kl_z_list = [torch.mean(kl(qzs, qzs_prior).sum(dim=1)) for qzs, qzs_prior in
                     zip(inference_outputs["qzs"], inference_outputs["qzs_prior"])]

        kl_z_dict = {'z_' + str(i+1): kl_z_list[i] for i in range(len(kl_z_list))}

        # classification metrics: CE, ACC, F1
        # print("GEX classification metrics: CE, ACC, F1 ")

        logits = self.classification_logits(inference_outputs)
        ce_loss_sum, accuracy, f1 = self.compute_clf_metrics(logits, cat_covs)
        ce_loss_mean = ce_loss_sum / len(range(self.zs_num))

        # total loss
        # print(f'reconst_loss_x_dict_acc is {reconst_loss_x_dict_acc}')
        # print(f'reconst_loss_x_dict is {reconst_loss_x_dict}')

        loss = reconst_loss_x + \
               reconst_loss_x_cf * cf_weight + \
               sum(kl_z_list) * kl_weight * beta + \
               ce_loss_sum * clf_weight + \
               reconst_loss_x_acc + \
               reconst_loss_x_cf_acc * cf_weight + \
               sum(kl_z_list_acc) * kl_weight * beta + \
               ce_loss_sum_acc * clf_weight #+ \ 
        
        # print(f'total loss is {loss}')
        # print(f'reconst_loss_x is {reconst_loss_x}')
        # print(f'reconst_loss_x_cf is {reconst_loss_x_cf}')
        # print(f'sum(kl_z_list) is {sum(kl_z_list)}')
        # print(f'ce_loss_sum is {ce_loss_sum}')
        # print(f'reconst_loss_x_acc is {reconst_loss_x_acc}')
        # print(f'reconst_loss_x_cf_acc is {reconst_loss_x_cf_acc}')
        # print(f'sum(kl_z_list_acc) is {sum(kl_z_list_acc)}')
        # print(f'ce_loss_sum_acc is {ce_loss_sum_acc}')


        loss_dict = {
            LOSS_KEYS.LOSS: loss,
            
            LOSS_KEYS.RECONST_LOSS_X: reconst_loss_x_dict,
            LOSS_KEYS.RECONST_LOSS_X_CF: reconst_loss_x_cf,
            LOSS_KEYS.KL_Z: kl_z_dict,
            LOSS_KEYS.CLASSIFICATION_LOSS: ce_loss_mean,
            LOSS_KEYS.ACCURACY: accuracy,
            LOSS_KEYS.F1: f1, 
            
            LOSS_KEYS.RECONST_LOSS_X_ACC: reconst_loss_x_dict_acc,
            LOSS_KEYS.RECONST_LOSS_X_CF_ACC: reconst_loss_x_cf_acc,
            LOSS_KEYS.KL_Z_ACC: kl_z_dict_acc,
            LOSS_KEYS.CLASSIFICATION_LOSS_ACC: ce_loss_mean_acc,
            LOSS_KEYS.ACCURACY_ACC: accuracy_acc,
            LOSS_KEYS.F1_ACC: f1_acc
        }

        return loss_dict
