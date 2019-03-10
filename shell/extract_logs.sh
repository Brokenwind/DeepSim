## 输入日志的目录,会自动从中提起出数据信息
#!/bin/bash

echo -n "Input the path of your log directory: "
read file_path

if [ -d "$file_path" ]; then
    cd "$file_path"
    for file in `ls ?*.log`
    do
        output_file="$file"_process.txt
        cat $file  | grep -E "^ -" | awk -F '[-:]' 'BEGIN {print "时间\t\t损失值\t交叉熵\t训练精度\t验证损失\t验证交叉熵\t验证精度"} {print $2,$4,$6,$8,$10,$12,$14}' > $output_file
    done
fi
