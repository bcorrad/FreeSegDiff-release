import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from functools import partial
from config import *

class AttentionHook:
    def __init__(self, unet, mapsSize:int=32, resolutions:list=[16]):
        self.__dict__.update(locals())
        self.att_modules = {
            # 64 x 64 blocks (not a big improvement in terms of clustering)
            'down_block0_0_64':unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2,
            'down_block0_1_64':unet.down_blocks[0].attentions[1].transformer_blocks[0].attn2,
            # 32 x 32 blocks (make clustering worse)
            'down_block1_0_32':unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2,
            'down_block1_1_32':unet.down_blocks[1].attentions[1].transformer_blocks[0].attn2,
            # 16 x 16 blocks (involved in object localization)
            'down_block2_0_16':unet.down_blocks[2].attentions[0].transformer_blocks[0].attn2,  # *
            'down_block2_1_16':unet.down_blocks[2].attentions[1].transformer_blocks[0].attn2,  # *
            # 16 x 16 blocks 
            'up_block1_0_16':unet.up_blocks[1].attentions[0].transformer_blocks[0].attn2,  # *
            'up_block1_1_16':unet.up_blocks[1].attentions[1].transformer_blocks[0].attn2,  # *
            'up_block1_2_16':unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2,  # *
            # 32 x 32 blocks (involved in searching for textures)
            'up_block2_0_32':unet.up_blocks[2].attentions[0].transformer_blocks[0].attn2,  # *
            'up_block2_1_32':unet.up_blocks[2].attentions[1].transformer_blocks[0].attn2,  # *
            'up_block2_2_32':unet.up_blocks[2].attentions[2].transformer_blocks[0].attn2,  # *
            # 64 x 64 blocks
            'up_block3_0_64':unet.up_blocks[3].attentions[0].transformer_blocks[0].attn2,
            'up_block3_1_64':unet.up_blocks[3].attentions[1].transformer_blocks[0].attn2,
            'up_block3_2_64':unet.up_blocks[3].attentions[2].transformer_blocks[0].attn2   
        }
        self.interpolationModeAttn = InterpolationMode.BILINEAR
        self.interpolationModeFeats = InterpolationMode.BILINEAR
        self.sdmBlocks =[
                            'down_block0_0_64', 
                            'down_block0_1_64',
                            'down_block1_0_32', 
                            'down_block1_1_32',
                            'down_block2_0_16', 
                            'down_block2_1_16', 
                            'up_block1_0_16', 
                            'up_block1_1_16', 
                            'up_block1_2_16',
                            'up_block2_0_32', 
                            'up_block2_1_32', 
                            'up_block2_2_32',
                            'up_block3_0_64', 
                            'up_block3_1_64', 
                            'up_block3_2_64',
                        ]
        self.resolutions = resolutions
        self.mapsSize = mapsSize
        self.selectBlock = [element_B for element_B in self.sdmBlocks for a in self.resolutions if str(a) in element_B]
        self.queries = {mod:None for mod in self.att_modules}
        for name, b in self.att_modules.items():
            b._forward_hooks = OrderedDict()
            def hook(mod, input, output, name):
                hidden_states = input[0]
                query = mod.to_q(hidden_states)
                query = mod.reshape_heads_to_batch_dim(query)
                self.queries[name] = query
            b.register_forward_hook(partial(hook, name=name))
        
    def set_text_embeddings(self, text_embeddings):
        self.keys = {key:mod.reshape_heads_to_batch_dim(mod.to_k(text_embeddings)) for key, mod in self.att_modules.items()}

    def computeAttentionMaps(self, concatRepresentation:bool=True):
        attention_scores_resized = list()
        attention_scores_dict = dict()
        for mod_name, mod in self.att_modules.items():
            if mod_name in self.selectBlock:
                key = self.keys[mod_name] # heads x classes x 160
                query = self.queries[mod_name] # a.k.a. feature maps heads x 256 x 160
                attention_scores = torch.einsum("b i d, b j d -> b i j", query, key) * mod.scale    # Q * K^T
                attention_scores = attention_scores.permute(0, 2, 1) 
                if attention_scores.shape[-1] == 16**2:
                    attention_scores = attention_scores.reshape(-1, 16, 16).cpu()  # 8 x classes x 16**2
                    attention_scores_dict[mod_name] = T.Resize(self.mapsSize, interpolation=self.interpolationModeAttn)(attention_scores)
                elif attention_scores.shape[-1] == 32**2:
                    attention_scores = attention_scores.reshape(-1, 32, 32).cpu()  # 8 x classes x 32**2
                    attention_scores_dict[mod_name] = T.Resize(self.mapsSize, interpolation=self.interpolationModeAttn)(attention_scores)
                elif attention_scores.shape[-1] == 64**2:
                    attention_scores = attention_scores.reshape(-1, 64, 64).cpu()  # 8 x classes x 64**2
                    attention_scores_dict[mod_name] = T.Resize(self.mapsSize, interpolation=self.interpolationModeAttn)(attention_scores)
                attention_scores_resized.append(T.Resize(self.mapsSize, interpolation=self.interpolationModeAttn)(attention_scores))
        if concatRepresentation:
            attention_scores_resized = np.concatenate(attention_scores_resized, axis=0)
            return attention_scores_resized, attention_scores_dict
        else:
            return attention_scores_dict

    def computeFeatureMaps(self, concatRepresentation:bool=True):
        feature_maps = list()
        feature_maps_dict = dict()
        # feature_maps_resized_16 = list()
        # feature_maps_resized_32 = list()
        # feature_maps_resized_64 = list()
        # hooks_res = []
        for mod_name, mod in self.att_modules.items():
            if mod_name in self.selectBlock:
                query = self.queries[mod_name]
                query = query.permute(0, 2, 1)
                if query.shape[-1] == 16**2:
                    query = query.reshape(-1, 16, 16).cpu() # (8, -1, 16, 16)
                    # hooks_res.append(16)
                    if not concatRepresentation:
                        feature_maps_dict[mod_name] = T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query)
                        # feature_maps_resized_16.append(T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query))
                elif query.shape[-1] == 32**2:
                    query = query.reshape(-1, 32, 32).cpu()
                    # hooks_res.append(32)
                    if not concatRepresentation:
                        feature_maps_dict[mod_name] = T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query)
                        # feature_maps_resized_32.append(T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query))
                elif query.shape[-1] == 64**2:
                    query = query.reshape(-1, 64, 64).cpu()
                    # hooks_res.append(64)
                    if not concatRepresentation:
                        feature_maps_dict[mod_name] = T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query)
                        # feature_maps_resized_64.append(T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query))
                feature_maps.append(T.Resize(self.mapsSize, interpolation=self.interpolationModeFeats)(query))
        if concatRepresentation:
            feature_maps = np.concatenate(feature_maps, axis=0)
            return feature_maps
        else:
            return feature_maps_dict