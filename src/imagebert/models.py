import logging
import random
import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertTokenizer,
    BertModel,
    BertPreTrainedModel
)
from transformers.modeling_bert import BertPreTrainingHeads

from typing import Tuple

default_logger=logging.getLogger(__name__)
default_logger.setLevel(level=logging.INFO)

BERT_MAX_SEQ_LENGTH=512 #BERTに入力するシーケンスの最大長

class ImageBertModel(BertModel):
    """
    ImageBERTのモデル
    """
    def __init__(
        self,
        config:BertConfig,
        add_pooling_layer:bool=True,
        roi_features_dim:int=1024,  #RoI特徴量の次元
        image_width:int=256,    #元画像の幅
        image_height:int=256,   #元画像の高さ
        logger:logging.Logger=default_logger):
        super().__init__(config,add_pooling_layer=add_pooling_layer)
        self.logger=logger
        self.roi_features_dim=roi_features_dim

        #FC層の作成
        #RoI関連のベクトルをBERTのhidden sizeに射影する。
        self.fc_roi_boxes=nn.Linear(5,config.hidden_size)
        self.fc_roi_features=nn.Linear(roi_features_dim,config.hidden_size)

        self.init_weights()

        #Position ID (トークンのインデックス)
        self.position_ids=torch.empty(BERT_MAX_SEQ_LENGTH,dtype=torch.long)
        for i in range(BERT_MAX_SEQ_LENGTH):
            self.position_ids[i]=i
        #テキストのToken Type IDは0
        self.text_token_type_ids=torch.zeros(BERT_MAX_SEQ_LENGTH,dtype=torch.long)
        #RoIのToken Type IDは1
        self.roi_token_type_ids=torch.ones(BERT_MAX_SEQ_LENGTH,dtype=torch.long)

        self.wh_tensor=torch.empty(5)  #(RoIの)Position Embedding作成に使用する。
        self.wh_tensor[0]=image_width
        self.wh_tensor[1]=image_height
        self.wh_tensor[2]=image_width
        self.wh_tensor[3]=image_height
        self.wh_tensor[4]=image_width*image_height

        #create_from_pretrained()でモデルを作成するか
        #set_sep_token_id()で明示的に設定すると有効になる。
        self.sep_token_id=None
        #forward()実行時に更新される。
        self.attention_mask=None

    @classmethod
    def create_from_pretrained(cls,pretrained_model_name_or_path,*model_args,**kwargs)->"ImageBertModel":
        """
        事前学習済みのモデルからパラメータを読み込み、ImageBERTのモデルを作成する。
        """
        model=ImageBertModel.from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)
        
        #[SEP]トークンのIDを取得する。
        tokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        model.sep_token_id=tokenizer.sep_token_id

        return model

    def set_sep_token_id(self,sep_token_id:int):
        self.sep_token_id=sep_token_id

    def get_attention_mask(self)->torch.Tensor:
        """
        Attention Maskを返す。
        Attention Maskはforward()実行時に更新されるので、
        forward()実行後にこのメソッドを使用すること。
        """
        return self.attention_mask

    def to(self,device:torch.device):
        super().to(device)

        self.fc_roi_boxes.to(device)
        self.fc_roi_features.to(device)
        self.position_ids=self.position_ids.to(device)
        self.text_token_type_ids=self.text_token_type_ids.to(device)
        self.roi_token_type_ids=self.roi_token_type_ids.to(device)
        self.wh_tensor=self.wh_tensor.to(device)

    def __create_embeddings(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor, #(N,max_num_rois,4)
        roi_features:torch.Tensor,   #(N,max_num_rois,roi_features_dim)
    )->torch.Tensor:
        """
        入力Embeddingを作成する。
        """
        word_embeddings=self.embeddings.word_embeddings
        position_embeddings=self.embeddings.position_embeddings
        token_type_ids_embeddings=self.embeddings.token_type_embeddings
        layer_norm=self.embeddings.LayerNorm
        dropout=self.embeddings.dropout

        device=input_ids.device

        batch_size=input_ids.size(0)
        max_num_rois=roi_boxes.size(1)

        v_position_embeddings=position_embeddings(self.position_ids)
        v_text_token_type_ids_embeddings=token_type_ids_embeddings(self.text_token_type_ids)
        v_roi_token_type_ids_embeddings=token_type_ids_embeddings(self.roi_token_type_ids)

        #=== テキストEmbeddingを作成する。===
        v_word_embeddings=word_embeddings(input_ids)    #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        #=== RoIのEmbeddingを作成する。 ===
        roi_features_embeddings=self.fc_roi_features(roi_features)
        #(N,max_num_rois,hidden_size)

        #RoIの座標から(RoIの)Position Embeddingを作成する。
        roi_position_embeddings=torch.empty(batch_size,max_num_rois,5).to(device)
        for i in range(batch_size):
            for j in range(max_num_rois):
                x_tl=roi_boxes[i,j,0]
                y_tl=roi_boxes[i,j,1]
                x_br=roi_boxes[i,j,2]
                y_br=roi_boxes[i,j,3]

                roi_position_embeddings[i,j,0]=x_tl
                roi_position_embeddings[i,j,1]=y_tl
                roi_position_embeddings[i,j,2]=x_br
                roi_position_embeddings[i,j,3]=y_br
                roi_position_embeddings[i,j,4]=(x_br-x_tl)*(y_br-y_tl)

        roi_position_embeddings=torch.div(roi_position_embeddings,self.wh_tensor)

        #RoIのPosition Embeddingを射影する。
        roi_position_embeddings=self.fc_roi_boxes(roi_position_embeddings)
        #(N,max_num_rois,hidden_size)

        roi_embeddings=roi_features_embeddings+roi_position_embeddings

        #=== テキストEmbeddingとRoI Embeddingを結合する。
        trunc_word_embeddings=v_word_embeddings[:,:BERT_MAX_SEQ_LENGTH-max_num_rois,:]

        text_roi_embeddings=torch.cat([trunc_word_embeddings,roi_embeddings],dim=1)
        #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        #[SEP]トークンのEmbeddingを入れる。
        if self.sep_token_id is not None:
            sep_input_ids=torch.Tensor([self.sep_token_id],dtype=torch.long).to(device)
            sep_embedding=word_embeddings(sep_input_ids)
            sep_embedding=torch.squeeze(sep_embedding)

            text_roi_embeddings[:,BERT_MAX_SEQ_LENGTH-max_num_rois-1]=sep_embedding.detach()
            text_roi_embeddings[:,-1]=sep_embedding.detach()

        trunc_text_token_type_ids_embeddings=v_text_token_type_ids_embeddings[:BERT_MAX_SEQ_LENGTH-max_num_rois]
        trunc_roi_token_type_ids_embeddings=v_roi_token_type_ids_embeddings[BERT_MAX_SEQ_LENGTH-max_num_rois:]
        v_token_type_ids_embeddings=torch.cat([trunc_text_token_type_ids_embeddings,trunc_roi_token_type_ids_embeddings],dim=0)
        #(BERT_MAX_SEQ_LENGTH,hidden_size)

        #Position EmbeddingとToken Type ID EmbeddingをExpandする。
        v_position_embeddings=v_position_embeddings.expand(batch_size,BERT_MAX_SEQ_LENGTH,-1)
        #(N,BERT_MAX_SEQ_LENGTH,hidden_size)
        v_token_type_ids_embeddings=v_token_type_ids_embeddings.expand(batch_size,BERT_MAX_SEQ_LENGTH,-1)
        #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        #最終的なEmbeddingはすべてを足したもの
        embeddings=text_roi_embeddings+v_position_embeddings+v_token_type_ids_embeddings
        embeddings=layer_norm(embeddings)
        embeddings=dropout(embeddings)  #(N,BERT_MAX_SEQ_LENGTH,hidden_size)

        return embeddings

    def __create_attention_mask(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor, #(N,max_num_rois,4)
    )->torch.Tensor:
        """
        attention_maskを作成する。
        """
        device=input_ids.device

        #テキスト部分
        text_attention_mask=(input_ids!=0).long().to(device)
        
        #RoI部分
        batch_size=roi_boxes.size(0)
        max_num_rois=roi_boxes.size(1)
        roi_attention_mask=torch.empty(batch_size,max_num_rois,dtype=torch.long).to(device)
        for i in range(batch_size):
            for j in range(max_num_rois):
                roi_box=roi_boxes[i,j]  #(4)

                #0ベクトルならそのRoIは存在しないので、attention_mask=0
                if torch.all(roi_box<1.0e-8):
                    roi_attention_mask[i,j]=0
                else:
                    roi_attention_mask[i,j]=1

        #テキスト部分とRoI部分のAttention Maskを結合する。
        text_attention_mask=text_attention_mask[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]
        attention_mask=torch.cat([text_attention_mask,roi_attention_mask],dim=1)

        return attention_mask

    def forward(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        token_type_ids:torch.Tensor=None,   #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor=None,    #(N,max_num_rois,4)
        roi_features:torch.Tensor=None,  #(N,max_num_rois,roi_features_dim)
        output_hidden_states:bool=None,
        return_dict:bool=None):
        """
        roi_boxesおよびroi_featuresを設定しない場合、
        普通のBERTモデル(テキストのみで動作する)になる。
        """
        device=input_ids.device

        ret=None
        #テキストのみで動作させる場合
        if roi_boxes is None or roi_features is None:
            attention_mask=(input_ids!=0).long().to(device)
            self.attention_mask=attention_mask

            ret=super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        #RoIのデータも入力する場合
        else:
            embeddings=self.__create_embeddings(input_ids,roi_boxes,roi_features)
            attention_mask=self.__create_attention_mask(input_ids,roi_boxes)
            self.attention_mask=attention_mask

            return_dict=return_dict if return_dict is not None else self.config.use_return_dict
            ret=super().forward(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return ret

class ImageBertForMultipleChoice(BertPreTrainedModel):
    """
    ImageBertModelのトップに全結合層をつけたもの
    BertForMultipleChoiceのImageBERT版
    """
    def __init__(self,config:BertConfig):
        super().__init__(config)

        self.imbert=ImageBertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,1)

        self.init_weights()

    def setup_image_bert(self,pretrained_model_name_or_path,*model_args,**kwargs):
        """
        パラメータを事前学習済みのモデルから読み込んでImageBERTのモデルを作成する。
        """
        self.imbert=ImageBertModel.create_from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)

    def to(self,device:torch.device):
        super().to(device)

        self.imbert.to(device)
        self.dropout.to(device)
        self.classifier.to(device)

    def forward(
        self,
        input_ids:torch.Tensor, #(N,num_choices,BERT_MAX_SEQ_LENGTH)
        labels:torch.Tensor,    #(N)
        token_type_ids:torch.Tensor=None,   #(N,num_choices,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor=None,    #(N,num_choices,max_num_rois,4)
        roi_features:torch.Tensor=None,  #(N,num_choices,max_num_rois,roi_features_dim)
        output_hidden_states:bool=None,
        return_dict:bool=None):
        num_choices=input_ids.size(1)
        input_ids=input_ids.view(-1,input_ids.size(-1)) #(N*num_choices,BERT_MAX_SEQ_LENGTH)
        roi_boxes=roi_boxes.view(-1,roi_boxes.size(-2),roi_boxes.size(-1)) #(N*num_choices,max_num_rois,4)
        roi_features=roi_features.view(-1,roi_features.size(-2),roi_features.size(-1))    #(N*num_choices,max_num_rois,roi_features_dim)

        outputs=self.imbert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            roi_boxes=roi_boxes,
            roi_features=roi_features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output=outputs[1]

        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        reshaped_logits=logits.view(-1,num_choices)

        criterion=nn.CrossEntropyLoss()
        loss=criterion(reshaped_logits,labels)

        if not return_dict:
            output=(reshaped_logits,)+outputs[2:]
            return ((loss,)+output) if loss is not None else output

        ret={
            "loss":loss,
            "logits":reshaped_logits,
            "hidden_states":outputs.hidden_states,
            "attentions":outputs.attentions,
        }
        return ret

class ImageBertForPreTraining(BertPreTrainedModel):
    """
    ImageBERTのPre-Trainingを行うためのクラス
    """
    def __init__(self,config:BertConfig):
        super().__init__(config)

        self.imbert=ImageBertModel(config)
        self.cls=BertPreTrainingHeads(config)

        self.init_weights()

        #setup_image_bert()でモデルをセットアップするか
        #set_mask_token_id()で明示的に設定すると有効になる。
        self.mask_token_id=None

    def setup_image_bert(self,pretrained_model_name_or_path,*model_args,**kwargs):
        """
        パラメータを事前学習済みのモデルから読み込んでImageBERTのモデルを作成する。
        """
        self.imbert=ImageBertModel.create_from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)

        tokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.mask_token_id=tokenizer.mask_token_id

    def set_mask_token_id(self,mask_token_id:int):
        self.mask_token_id=mask_token_id

    def __create_masked_token_ids_and_masked_lm_labels(
        self,
        input_ids:torch.Tensor,
        max_num_rois:int)->Tuple[torch.Tensor,torch.Tensor]:
        """
        返されるTensorはRoI部分を考慮してTruncateされたもの
        """
        device=input_ids.device
        batch_size=input_ids.size(0)

        masked_token_ids=input_ids.detach().clone()
        masked_token_ids=masked_token_ids[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]
        masked_lm_labels=torch.ones(batch_size,BERT_MAX_SEQ_LENGTH-max_num_rois,dtype=torch.long)*(-100)
        masked_lm_labels=masked_lm_labels.to(device)

        #全体の15%のトークンがマスクされる。
        mask_rnds=torch.rand(batch_size,BERT_MAX_SEQ_LENGTH-max_num_rois).to(device)
        mask_flags=(mask_rnds<0.15).to(device)
        for i in range(batch_size):
            for j in range(BERT_MAX_SEQ_LENGTH-max_num_rois):
                if mask_flags[i,j]==False:
                    continue

                prob_rnd=random.random()
                #10%の確率で変更しない。
                if prob_rnd<0.1:
                    pass
                #10%の確率でランダムなトークンに変更する。
                elif prob_rnd<0.2:
                    masked_lm_labels[i,j]=input_ids[i,j]
                    masked_token_ids[i,j]=random.randrange(self.config.vocab_size)
                #80%の確率で[MASK]トークンに変更する。
                else:
                    masked_lm_labels[i,j]=input_ids[i,j]
                    masked_token_ids[i,j]=self.mask_token_id

        return masked_token_ids,masked_lm_labels

    def __create_masked_roi_labels_and_masked_oc_labels(
        self,
        roi_labels:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        device=roi_labels.device
        batch_size=roi_labels.size(0)
        max_num_rois=roi_labels.size(1)

        masked_roi_labels=roi_labels.detach().clone()
        masked_oc_labels=torch.ones(batch_size,max_num_rois,dtype=torch.long)*(-100)
        masked_oc_labels=masked_oc_labels.to(device)

        #全体の15%のトークンがマスクされる。
        mask_rnds=torch.rand(batch_size,max_num_rois).to(device)
        mask_flags=(mask_rnds<0.15).to(device)
        for i in range(batch_size):
            for j in range(max_num_rois):
                if mask_flags[i,j]==False:
                    continue

                prob_rnd=random.random()
                #10%の確率で変更しない。
                if prob_rnd<0.1:
                    pass
                #90%の確率で0に変更する。
                else:
                    masked_oc_labels[i,j]=roi_labels[i,j]
                    masked_roi_labels[i,j]=0

        return masked_roi_labels,masked_oc_labels

    def __create_negative_samples(
        self,
        input_ids:torch.Tensor, #(N,BERT_SEQ_MAX_LENGTH)
        roi_boxes:torch.Tensor, #(N,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,max_num_rois,roi_features_dim)
        roi_labels:torch.Tensor,    #(N,max_num_rois)
        create_negative_prob:float=0.5):   
        """
        Image-Text Matching (ITM)を行うための負例を作成する。

        create_negative_probで指定された確率でバッチに含まれるサンプルを負例にする。
        たとえばcreate_negative_prob=0.5なら、
        50%の確率でこのバッチの全サンプルはテキストとRoIが対応しない負例になり、
        50%の確率で何も変更されない。

        負例を作成する場合には、n番目のサンプルのRoIデータを(n+1)%N番目のRoIデータに変更する。

        作成された例はDict形式で返される。
        """
        if random.random()>create_negative_prob:
            ret={
                "input_ids":input_ids,
                "roi_boxes":roi_boxes,
                "roi_features":roi_features,
                "roi_labels":roi_labels,
                "is_negative":False
            }
            return ret

        batch_size=input_ids.size(0)

        sample_0={
            "roi_boxes":roi_boxes[0].detach().clone(),
            "roi_features":roi_features[0].detach().clone(),
            "roi_labels":roi_labels[0].detach().clone()
        }
        for i in range(batch_size):
            if i==batch_size-1:
                roi_boxes[i]=sample_0["roi_boxes"]
                roi_features[i]=sample_0["roi_features"]
                roi_labels[i]=sample_0["roi_labels"]
            else:
                roi_boxes[i]=roi_boxes[i+1]
                roi_features[i]=roi_features[i+1]
                roi_labels[i]=roi_labels[i+1]

        ret={
            "input_ids":input_ids,
            "roi_boxes":roi_boxes,
            "roi_features":roi_features,
            "roi_labels":roi_labels,
            "is_negative":True
        }
        return ret

    def forward(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor,    #(N,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,max_num_rois,roi_features_dim)
        roi_labels:torch.Tensor,    #(N,max_num_rois)
        output_hidden_states:bool=None,
        return_dict:bool=None):
        """
        roi_labelsはFaster R-CNNで検出されたRoIのクラス
        """
        device=input_ids.device

        batch_size=roi_features.size(0)
        max_num_rois=roi_boxes.size(1)

        #入力サンプルの作成
        samples=self.__create_negative_samples(input_ids,roi_boxes,roi_features,roi_labels)
        input_ids=samples["input_ids"]
        roi_boxes=samples["roi_boxes"]
        roi_features=samples["roi_features"]
        roi_labels=samples["roi_labels"]
        is_negative=samples["is_negative"]

        itm_labels=None
        if is_negative:
            itm_labels=torch.ones(batch_size,dtype=torch.long).to(device)
        else:
            itm_labels=torch.zeros(batch_size,dtype=torch.long).to(device)

        #Masked Language Modeling (MLM)およびMasked Object Classification (MOC)用の入力の作成
        masked_token_ids,masked_lm_labels=self.__create_masked_token_ids_and_masked_lm_labels(input_ids,max_num_rois)
        masked_roi_labels,masked_oc_labels=self.__create_masked_roi_labels_and_masked_oc_labels(roi_labels)

        masked_input_ids=torch.cat([masked_token_ids,masked_roi_labels],dim=1)
        masked_lm_oc_labels=torch.cat([masked_lm_labels,masked_oc_labels],dim=1)

        outputs=self.imbert(
            input_ids=masked_input_ids,
            roi_boxes=roi_boxes,
            roi_features=roi_features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        #Attention Maskが0の部分はLoss計算時に無視する。
        attention_mask=self.imbert.get_attention_mask() #(N,BERT_MAX_SEQ_LENGTH)
        masked_lm_oc_labels[attention_mask==0]=-100

        #各種Lossの計算
        criterion_ce=nn.CrossEntropyLoss()
        criterion_mse=nn.MSELoss()

        #Masked Language Modeling (MLM)
        #Masked Object Classification (MOC)
        #Image-Text Matching (ITM)
        sequence_output,pooled_output=outputs[:2]
        prediction_scores,seq_relationship_score=self.cls(sequence_output,pooled_output)

        masked_lm_oc_loss=criterion_ce(prediction_scores.view(-1,self.config.vocab_size),masked_lm_oc_labels.view(-1))
        itm_loss=criterion_ce(seq_relationship_score.view(-1,2),itm_labels.view(-1))

        #Masked Region Feature Regression (MRFR)
        mrfr_loss=0
        for i in range(batch_size):
            for j in range(BERT_MAX_SEQ_LENGTH-max_num_rois,BERT_MAX_SEQ_LENGTH):
                #マスクされているRoIトークンについてLossを計算する。
                if masked_lm_oc_labels[i,j]!=-100:
                    input=sequence_output[i,j]
                    target=roi_features[i,j-(BERT_MAX_SEQ_LENGTH-max_num_rois)]

                    mrfr_loss+=criterion_mse(input,target)

        total_loss=masked_lm_oc_loss+itm_loss+mrfr_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        ret={
            "loss":total_loss,
            "prediction_logits":prediction_scores,
            "seq_relationship_logits":seq_relationship_score,
            "hidden_states":outputs.hidden_states,
            "attentions":outputs.attentions
        }
        return ret
