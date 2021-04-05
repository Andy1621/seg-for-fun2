echo "=====Start preparing dataset====="

# 解压压缩包并划分训练集与验证集
bash ./exp/create_txt.sh ./raw_data ./data/rs_data

# 划分类4类5正负样本, 原数据类4类5重采样, 生成类0类3类4二分类数据
# python3 ./tools/generate_my_dataset.py ./data/rs_data/train_data


echo "=====Preparing dataset over====="