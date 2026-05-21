import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
)

# Assuming ConfigsNoor contains a dictionary with configurations
from ModelsConfigsNew import modelsConfigDict


class MultiTaskModel(nn.Module):
    def __init__(self, model_config, num_labels,num_decades, device,experType="all", pretrained=True):
        """
        Initialize the multitask model with the given backbone.

        Args:
            model_config (str): The Model configs  (e.g., "VGG", "ResNet", "DenseNet", "ViT", "Swin").
            num_labels (int): Number of classes for classification.
            device (str): Device to use ("cuda" or "cpu").
            experType (str): kind of model (type-1-- > (classification head for type --- regression head to deacde) )  type-2 -> ( classification head for both type + decade) 
            all --> (type,year,decade) classifcation head for type , regression head for both year and deacde
            pretrained (bool): Whether to use pretrained weights for the backbone.
        """
        super(MultiTaskModel, self).__init__()
        self.model_config = model_config
        self.num_labels = num_labels
        self.num_decades = num_decades
        self.experType = experType
        self.device = device
        self.pretrained = pretrained
        self.model = self.create_model(self.model_config,self.num_labels)
        self.company, self.model_name, self.version = self.parse_config_name(self.model_config)
        print(f"Company: {self.company}, Model: {self.model_name}, Version: {self.version}")
        self.classification_head_type , self.regression_classification_head_decade , self.regression_head_year  = self.modify_model()

    def create_model(self, config_name, num_labels):
        config = AutoConfig.from_pretrained(
            config_name,
            num_labels=num_labels,
            finetuning_task="image-classification"
        )
        model = AutoModelForImageClassification.from_pretrained(
            config_name,
            config=config,
            revision='main',
            ignore_mismatched_sizes=True
        )
        return model

    def parse_config_name(self,config_name):
        """
        Parse the configuration name to extract company, model, and version.

        Args:
            config_name (str): A string containing the full configuration name.

        Returns:
            tuple: A tuple containing the company, model, and version.
        """
        # Split the string by '/' to separate the company and the rest
        parts = config_name.split('/')
        company = parts[0]
        # Further split by '-' to separate model and version
        model_version = parts[1].split('-')
        #model = '-'.join(model_version[:-1])  # Join all parts except the last as the model name

        if company == "apple":
            model = model_version[0]
            version = model_version[-1]
            if version != model_version[1] :
                version = f"{version}_{model_version[1]}"
            print("aaaaaaa ",model,version)
        else :
           model = model_version[0]
           version = model_version[1]  # The last part is the version
        return company, model, version

    def modify_model(self):
        """
        Modify the model to add classification and regression heads.
        """
        # self.model.classifier = None
        if self.model_name == "swin":
            feature_dim = self.model.swin.encoder.layers[-1].blocks[-1].layernorm_before.normalized_shape[0]
        elif self.model_name == "beit":
            feature_dim = self.model.beit.config.hidden_size
        elif self.model_name == "mobilevit":
            feature_dim = self.model.classifier.in_features
        elif self.model_name == "vit":
            feature_dim = self.model.classifier.in_features
        elif self.model_name == "convnext":
            feature_dim = self.model.classifier.in_features
        elif self.model_name == "focalnet":
            feature_dim = self.model.classifier.in_features

        else:
            raise ValueError(f"Unsupported model or version: {self.model_name,self.version}")


        self.model.classifier = None
        if self.experType == "type-1":
          # Multi-task heads
          classification_head_type = nn.Linear(feature_dim, self.num_labels)
          regression_head_deacde = nn.Linear(feature_dim, 1)
          return classification_head_type , regression_head_deacde , None
        elif self.experType == "type-2":
          # Multi-task heads
          classification_head_type = nn.Linear(feature_dim, self.num_labels)
          classification_head_decade = nn.Linear(feature_dim, self.num_decades)          
          return  classification_head_type,classification_head_decade , None
        elif self.experType == "all":
          # Multi-task heads
          classification_head_type = nn.Linear(feature_dim, self.num_labels)
          regression_head_year = nn.Linear(feature_dim, 1)
          regression_head_decade = nn.Linear(feature_dim, 1)           
          return  classification_head_type ,regression_head_decade , regression_head_year         
          
        return None,None,None

    def forward(self, x):
        x = x.to(self.device)  # Make sure input tensor is on the same device as model
        try:
            if self.model_name == "swin":
                features = self.model.swin(x).pooler_output
            if self.model_name == "beit":
                features = self.model.beit(x).pooler_output
            if self.model_name == "mobilevit":
                features = self.model.mobilevit(x).pooler_output
            if self.model_name == "convnext":
                features = self.model.convnext(x).pooler_output
            if self.model_name == "focalnet":
                features = self.model.focalnet(x).pooler_output
            if self.model_name == "vit":
                features = self.model.vit(x).last_hidden_state
                gap = nn.AdaptiveAvgPool1d(1)  # Pooling to a single value for each feature dimension
                pooled_output = gap(features.permute(0, 2, 1))  # Permute to [batch_size, feature_dim, num_patches]
                features = pooled_output.squeeze(-1)  # Remove the last dimension (num_patches)

            # print(f"Extracted features shape: {features.shape}")  # Debugging output.
            cls_logits_type = self.classification_head_type(features)
            reg_output_deacde = self.regression_classification_head_decade(features)
            
            if self.experType == "all" :
              reg_output_year = self.regression_head_year(features)     
              return cls_logits_type, reg_output_deacde , reg_output_year

            return cls_logits_type, reg_output_deacde , None
        except Exception as e:
            print(f"Error during forward pass of {self.model_name,self.version}: {e}")
            raise


if __name__ == '__main__':
    num_labels = 14
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for company in modelsConfigDict.keys():
        for cfgModel in modelsConfigDict[company]:
            if 'mobilevit' not in cfgModel:
                continue
            dummy_input = torch.randn(4, 3, 224, 224).to(device)
            print(cfgModel)
            model = MultiTaskModel(cfgModel, num_labels=num_labels,num_decades = 10 ,device=device, pretrained=True)
            model.to(device)
            # print(model)
            cls_logits, reg_output , reg_output2 = model(dummy_input)
            print(cls_logits,reg_output)
            print(f"Classification logits shape: {cls_logits.shape}")  # Expected: (4, num_labels)
            print(f"Regression output shape: {reg_output.shape}")  # Expected: (4, 1)
            print(f"Regression reg_output2 shape: {reg_output2.shape}")  # Expected: (4, 1)
