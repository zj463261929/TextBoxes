#coding=utf-8

#创建本地仓库：
git init   #把这个目录变成Git可以管理的仓库
git add .  #告诉Git，把文件添加到仓库
git commit -m "readme"  #告诉Git，把文件提交到仓库： -m后面输入的是本次提交的说明
git status  #

git remote add origin git@github.com:zj463261929/TextBoxes.git  #在本地关联的就是我的远程库
#git remote add origin https://github.com/zj463261929/trietree.git

#git push -u origin master  #第一次推送master分支的所有内容；
git push -f origin master  #第一次推送master分支的所有内容；
#git push origin master  # 推送最新修改, 把本地master分支的最新修改推送至GitHub  

#http://www.cnblogs.com/wangmingshun/p/5424767.html

#git clone git@github.com:zj463261929/trietree.git
#git remote add origin git@github.com:tongpi/basicOCR.git



