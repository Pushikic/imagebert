import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertTokenizer,
    BertModel
)

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
    ):
        super().__init__(config,add_pooling_layer=add_pooling_layer)
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

        device=self.fc_roi_boxes.weight.device
        input_ids=input_ids.to(device)
        roi_boxes=roi_boxes.to(device)
        roi_features=roi_features.to(device)
        self.position_ids=self.position_ids.to(device)
        self.text_token_type_ids=self.text_token_type_ids.to(device)
        self.roi_token_type_ids=self.roi_token_type_ids.to(device)
        self.wh_tensor=self.wh_tensor.to(device)

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
            sep_input_ids=torch.tensor([self.sep_token_id]).to(device)
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
        device=self.fc_roi_boxes.weight.device
        input_ids=input_ids.to(device)
        roi_boxes=roi_boxes.to(device)

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
        device=self.fc_roi_boxes.weight.device
        input_ids=input_ids.to(device)
        if token_type_ids is not None:
            token_type_ids=token_type_ids.to(device)
        if roi_boxes is not None:
            roi_boxes=roi_boxes.to(device)
        if roi_features is not None:
            roi_features=roi_features.to(device)

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
