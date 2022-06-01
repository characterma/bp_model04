2020-08-13
按照  2020-08-13-血压新模型方案.docx 的步骤, 整理模型

目录结构:
data --|
       --| raw_data  原始数据,包括配合康康血压计采集的ppg数据, 从数据库拉取的客服记录的数据
       --| train_data
       --| test_data


data_process --|
                --| old_data_process


doc --|
       --| 2020-08-13-血压新模型方案.docx






2020-08-21

1 在 bp_model02 的基础上, 把之前的脚本进行封装, 成一键调用,把训练的模型直接保存, 把多余的预测, 测试部分删除, 使保存的模型, 客户端调用的时候可以直接使用
2 封装 grpc



2021-03-05
增加钉钉预警消息提醒功能

1 安装包 pip install DingtalkChatbot -i  http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
2 暂停端口 50057
3 utils.py 替换掉 bp_model/utils/utils.py
4 anbp_server.py 替换掉 bp_model/grpc/server/anbp_server.py
5 启动服务 python grpc/server/anbp_server.py