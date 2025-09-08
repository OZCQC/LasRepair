import torch.nn as nn


class CustomT5Model(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Add your custom processing layer here
        self.processing_layer = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(base_model.config.hidden_size, base_model.config.vocab_size)
        )
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,  # We need hidden states for our processing
            **kwargs
        )
        
        if labels is not None:
            # During training, use base model outputs directly
            return outputs
        else:
            # During inference, we'll use the processed logits
            last_hidden_states = outputs.hidden_states[-1]
            processed_logits = self.processing_layer(last_hidden_states)
            outputs.logits = processed_logits
            return outputs
    
    # there are some bugs, super.__init__() does not work, so add this function.
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        # Delegate generation to the base model
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )