"""
RDAQ: DETR meets Refinement-Driven Adaptive Querying for Dense Aerial Imagery
Copyright (c) 2025 The RDAQ Authors. All Rights Reserved.
"""

import torch   
import torch.nn as nn
import torch.nn.functional as F
 
import torchvision    
   
from ..core import register
    

__all__ = ['PostProcessor', 'DQPostProcessor']
   

def mod(a, b):   
    out = a - a // b * b
    return out 

@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',     
        'use_focal_loss',     
        'num_top_queries',
        'remap_mscoco_category' 
    ]     

    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True,  
        num_top_queries=300,  
        remap_mscoco_category=False  
    ) -> None:
        super().__init__() 
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False  
   
    def extra_repr(self) -> str:   
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'    

    def forward(self, outputs, orig_target_sizes: torch.Tensor): 
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        if 'num_queries_list' in outputs:   
            num_queries_list = outputs['num_queries_list']   
            for i, qn in enumerate(num_queries_list):
                if self.use_focal_loss:
                    logits[i, qn:] = -10000
                else:
                    logits[i, qn:] = 0 
   
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')    
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1) 
     
        if self.use_focal_loss:   
            scores = F.sigmoid(logits)  
            scores, index = torch.topk(scores.flatten(1), max(num_queries_list), dim=-1) 
            labels = index % self.num_classes    
            index = index // self.num_classes    
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else: 
            scores = F.softmax(logits, dim=-1)[:, :, :-1]    
            scores, labels = scores.max(dim=-1)  
            if scores.shape[1] > max(num_queries_list):  
                scores, index = torch.topk(scores, max(num_queries_list), dim=-1)     
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))   
        if self.deploy_mode:    
            return labels, boxes, scores 

        # 如果需要映射 COCO 类别索引    
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category   
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)   
     
        results = []
        for lab, box, sco, qn in zip(labels, boxes, scores, num_queries_list):
            # print(qn, lab.size(0))    
            result = dict(labels=lab[:qn], boxes=box[:qn], scores=sco[:qn])
            results.append(result)    

        return results    

    def deploy(self):  
        self.eval()    
        self.deploy_mode = True
        return self