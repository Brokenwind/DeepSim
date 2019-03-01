# DeepSim

### 下载并提取语料
- 下载语料库  
  中文语料库中，质量高而又容易获取的语料库，应该就是维基百科的中文语料，而且维基百科相当厚道，每个月都把所有条目都打包一次，下载的时候下载最新的就可以了。  
  [下载地址](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2)  
- 提取语料  
  但是使用维基百科语料还是有一定门槛的，直接下载下来的维基百科语料是一个带有诸多html和markdown标记的文本压缩包，基本不能直接使用，所以需要从中提取中有用信息。  
  这里使用的是WikiExtractor,使用方法也很简单：
  * 先下载工具代码 
  ```
  git clone https://github.com/attardi/wikiextractor.git
  ```  
  * 然后安装  
  ```
  sudo python3 setup.py install
  ```
  * 使用方法  
  ```
  WikiExtractor.py -b 500M -o zhwiki zhwiki-latest-pages-articles.xml.bz2
  ```
    > -o 指的是输出文件夹路径
    > -b 指的是提取后语料语料文件的切分大小
- 繁体转简体  
  维基百科上有很多繁体文章，需要将其转换为简体，这里我们使用opencc工具。
  使用说明地址：[opencc](http://blog.sina.com.cn/s/blog_703521020102zb5v.html)
  * 安装方式  
  ```
  # 安装命令行工具
  sudo apt-get install opencc
  # 安装Python接口
  sudo pip install opencc
  ```
  * 使用方式  
    - 命令行使用
    ```
    opencc -i wiki_00 -o zh_wiki_00 -c zht2zhs.ini
    ```
    - 代码中使用  
    ```
    from opencc import OpenCC
    # t2s 繁体变简体
    opencc1 = OpenCC("t2s")
    res = opencc1.convert(s).strip()
    ```
