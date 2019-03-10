## 输入日志的目录,会自动从中提起出数据信息
#!/bin/bash

echo -n "Input the path of your log directory: "
read file_path

if [ -d "$file_path" ]; then
    cd "$file_path"
    for file in `ls ?*.log`
    do
        output_file="$file"_process.csv
        cat $file  | grep -E "^ -" | awk -F '[-:]' 'BEGIN {print " 时间 , 损失值 , 交叉熵 , 训练精度 ,验证损失,验证交叉熵,验证精度"} {printf("%s,%s,%s,%s,%s,%s,%s\n",$2,$4,$6,$8,$10,$12,$14)}' > $output_file
    done
fi
