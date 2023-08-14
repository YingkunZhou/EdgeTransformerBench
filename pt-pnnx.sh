./pnnx efficientformerv2_s0.pt inputshape=[1,3,224,224]
./pnnx efficientformerv2_s1.pt inputshape=[1,3,224,224]
./pnnx efficientformerv2_s2.pt inputshape=[1,3,224,224]

# layer load_model 35 normalize_16 failed
./pnnx SwiftFormer_XS.pt inputshape=[1,3,224,224]
./pnnx SwiftFormer_S.pt  inputshape=[1,3,224,224]
./pnnx SwiftFormer_L1.pt inputshape=[1,3,224,224]

# [1]    720797 segmentation fault (core dumped)  ./ncnn_perf --only-test EMO_1M
./pnnx EMO_1M.pt inputshape=[1,3,224,224]
./pnnx EMO_2M.pt inputshape=[1,3,224,224]
./pnnx EMO_5M.pt inputshape=[1,3,224,224]
./pnnx EMO_6M.pt inputshape=[1,3,224,224]

# layer load_model 74 normalize_30 failed
./pnnx edgenext_xx_small.pt inputshape=[1,3,256,256]
./pnnx edgenext_x_small.pt  inputshape=[1,3,256,256]
./pnnx edgenext_small.pt    inputshape=[1,3,256,256]

./pnnx mobilevitv2_050.pt inputshape=[1,3,256,256]
./pnnx mobilevitv2_075.pt inputshape=[1,3,256,256]
./pnnx mobilevitv2_100.pt inputshape=[1,3,256,256]
./pnnx mobilevitv2_125.pt inputshape=[1,3,256,256]
./pnnx mobilevitv2_150.pt inputshape=[1,3,256,256]
./pnnx mobilevitv2_175.pt inputshape=[1,3,256,256]
./pnnx mobilevitv2_200.pt inputshape=[1,3,256,256]

# (index: 999,  score: -nan), (index: 998,  score: -nan), (index: 997,  score: -nan),
./pnnx mobilevit_xx_small.pt inputshape=[1,3,256,256]
./pnnx mobilevit_x_small.pt  inputshape=[1,3,256,256]
./pnnx mobilevit_small.pt    inputshape=[1,3,256,256]

# layer torch.flatten not exists or registered
./pnnx LeViT_128S.pt inputshape=[1,3,224,224]
./pnnx LeViT_128.pt  inputshape=[1,3,224,224]
./pnnx LeViT_192.pt  inputshape=[1,3,224,224]
./pnnx LeViT_256.pt  inputshape=[1,3,224,224]

./pnnx resnet50.pt inputshape=[1,3,224,224]
./pnnx mobilenetv3_large_100.pt inputshape=[1,3,224,224]
./pnnx tf_efficientnetv2_b0.pt  inputshape=[1,3,224,224]
./pnnx tf_efficientnetv2_b1.pt  inputshape=[1,3,240,240]
./pnnx tf_efficientnetv2_b2.pt  inputshape=[1,3,260,260]
./pnnx tf_efficientnetv2_b3.pt  inputshape=[1,3,300,300]