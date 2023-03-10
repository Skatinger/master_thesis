from transformers import FillMaskPipeline
from typing import Dict
from transformers.pipelines.base import GenericTensor

class FillMaskPipelineWithTruncation(FillMaskPipeline):
    """Overwrite the fill-mask pipeline to allow for truncation of the input text and parsing preprocessing parameters.
    Args:
        same arguments as FillMaskPipeline, extended with truncation and padding parameters for the tokenizer.
    Returns:
        same as FillMaskPipeline
    """    
    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters) -> Dict[str, GenericTensor]:
        """Overwrites the preprocess method of the fill-mask pipeline to allow for truncation of the input text."""
        preprocess_parameters["truncation"] = True
        preprocess_parameters["padding"] = 'max_length'
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, truncation=True, padding='max_length')
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs