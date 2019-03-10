## 输入日志的目录,会自动从中提起出数据信息
#!/bin/bash

echo -n "Input the path of your log directory: "
read file_path

if [ -d "$file_path" ]; then
    cd "$file_path"
    for file in `ls ?*.log`
    do
        name=${file%.*}
        output_file="$file"_process.csv
        cat $file  | grep -E "^ -" | awk -F '[-:]' 'BEGIN {print "轮次, 时间 , 损失值 , 交叉熵 , 训练精度 ,验证损失,验证交叉熵,验证精度,类型"} {printf("%s,%s,%s,%s,%s,%s,%s,%s,%s\n",NR,$2,$4,$6,$8,$10,$12,$14,"'$name'")}' > $output_file
    done
fi
