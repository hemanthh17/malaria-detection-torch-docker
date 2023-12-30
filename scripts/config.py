class Config:
    train_pth='C:/Users/Hemanth/Desktop/Data Analytics analyticvidya/pytorch malaria detection-docker/Dataset/Train'
    test_pth='C:/Users/Hemanth/Desktop/Data Analytics analyticvidya/pytorch malaria detection-docker/Dataset/Test'
    model_pth='C:/Users/Hemanth/Desktop/Data Analytics analyticvidya/pytorch malaria detection-docker/models'
    model_name='vit_model.pth'
    img_shape=(224,224)
    trans_model='vit_base_patch32_clip_224'
    epochs=10
    bs=32
    lr=1e-3
    data_stat=True
