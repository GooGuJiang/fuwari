---
title: 2025年软件系统安全赛 5G消息 流量分析
published: 2025-03-23
description: ''
image: ''
tags: ['CTF','Wireshark',]
category: 'CTF Writeup'
draft: false 
lang: ''
---

# 碎碎念

这题7个小时都没弄出来太可惜了，真的很简单，主要对Wireshark软件不熟悉，唉~

![image-20250323235245637](https://gu-blog-img.gmoe.cc/20250323235245688.png)

# 题目分析

拿到数据包往下翻发现SIP数据包通过流追踪发现是 Bob 给 Alice发的5G消息![image-20250323235951717](https://gu-blog-img.gmoe.cc/20250323235951855.png)

这第一条消息是

```
Alice, I am Bob. I stole the sslkeylog file, this is crazy.
Alice，我是Bob。我偷了sslkeylog文件，这太疯狂了。
```

继续往下翻会发现 Bob 给 Alice发的`sslkeylog`内容,分别发了3条信息，这就是`sslkeylog`

SSLKey格式 `<标签> <会话标识> <密钥值>`

```
CLIENT_HANDSHAKE_TRAFFIC_SECRET 9745a631db0b9b715f18a55220e17c88fdf3389c0ee899cfcc45faa8696462c1 a98fab3039737579a50e2b3d0bbaba7c9fcf6881d26ccf15890b06d723ba605f096dbe448cd9dcc6cf4ef5c82d187bd0
SERVER_HANDSHAKE_TRAFFIC_SECRET 9745a631db0b9b715f18a55220e17c88fdf3389c0ee899cfcc45faa8696462c1 994da7436ac3193aff9c2ebaa3c072ea2c5b704683928e9f6e24d183e7e530386c1dcd186b9286f98249b4dc90d8b795
CLIENT_TRAFFIC_SECRET_0 9745a631db0b9b715f18a55220e17c88fdf3389c0ee899cfcc45faa8696462c1 646306cb35d94f23e125225dc3d3c727df65b6fcec4c6cd77b6f8e2ff36d48e2b7e92e8f9188597c961866b3b667f405
SERVER_TRAFFIC_SECRET_0 9745a631db0b9b715f18a55220e17c88fdf3389c0ee899cfcc45faa8696462c1 1fbf7c07ca88c7c91be9cce4c9051f2f4bd7fb9714920661d026119ebab458db8637089348dd5a92dc75633bdcf43630
EXPORTER_SECRET 9745a631db0b9b715f18a55220e17c88fdf3389c0ee899cfcc45faa8696462c1 31882156a3212a425590ce171cb78068ee63e7358b587fed472d45d67ea567d98a079c84867a18665732cf0bfe18f0b0
```

然后通过`Wireshark`加载该`sslkeylog`即可解密数据包中tls流量，就会发现一张图片，里面就是flag

# 解题步骤

### 1.打开分析数据包

![image-20250324001051851](https://gu-blog-img.gmoe.cc/20250324001052024.png)

Bob在这个时候给Alice发送消息

### 2.跟踪该流获

跟踪该流即可拿到Bob给Alice发送的sslkeylog文件，然后复制保存成txt文件

![image-20250324001544294](https://gu-blog-img.gmoe.cc/20250324001544362.png)

### 3.Wireshark解密TLS流量

打开Wireshark的首选项

![image-20250324001819724](https://gu-blog-img.gmoe.cc/20250324001819820.png)

展开`Protocols`选项找到`TLS`设置

![image-20250324001907297](https://gu-blog-img.gmoe.cc/20250324001907359.png)

设置`SSLKeyLog`文件

![image-20250324002756976](https://gu-blog-img.gmoe.cc/20250324002757077.png)

### 4.筛选TLS流量

在这里就能看到多出来两个HTTP数据包，一个是一张图片

![image-20250324002920517](https://gu-blog-img.gmoe.cc/20250324002920595.png)

### 5.导出HTTP（PNG）这个数据包内传输的内容

展开`Portable...`导出分组字节流为图片即可获得flag

![image-20250324003308986](https://gu-blog-img.gmoe.cc/20250324003309068.png)

# 获得的FLAG

![image-20250324003413161](https://gu-blog-img.gmoe.cc/20250324003413209.png)

