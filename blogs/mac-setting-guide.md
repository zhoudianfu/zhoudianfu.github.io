---
layout: page
permalink: /blogs/mac-setting-guide/index.html
title: mac-setting-guide
---

## 从零配置mac开发环境

<br>最近换了一台macbook air m2，24+512版本。国补下来5199，还是觉得性价比不错。本想是m4的16+512，但想想没有必要，m2够用了哈哈哈(希望后面不打脸)。来写写自己的简单配置，希望对你有所帮助。

<br><u>如果想购入mac，建议大家是有明确的需求才买的。</u>

mac优点：

- 续航长
- 很轻
- 写代码和文档，连连服务器很丝滑，体验很好。

<br>配置环境视频与文档：

- 【新手从零使用mac】https://space.bilibili.com/49574614/lists/278381?type=season
- 【Mac 开发环境基础配置】 https://www.bilibili.com/video/BV1ggJszHE3R/?share_source=copy_web&vd_source=783046dd26b6d8ed3ae12d74958b0584
- 【如何像极客一样使用你的Mac？分享下我的Mac工作流】 https://www.bilibili.com/video/BV1Yq8tzdEPU/?share_source=copy_web&vd_source=783046dd26b6d8ed3ae12d74958b0584
- https://arthals.ink/blog/initialize-mac
- https://sourabhbajaj.com/mac-setup/

按照上面顺序配置即可，大部分使用和环境都ok。

<br>踩坑点

- 配置homebrew记得设置终端代理，直接开梯子不会使得终端也代理。在.zshrc添加`export http_proxy="http://127.0.0.1:1082"
  export https_proxy="http://127.0.0.1:1082"`，具体代理查看你梯子的代理端口。
- openinterminal和cleanclip一般是必用软件，需要安装
- 改快捷键，比如截屏的改自己设置的简单一点
- 一些需要pro和会员的软件可以去macwk、mac618、xmac、appstorerent等等下载

<br>
<center>
<img src="/blogs/mac-setting-guide.assets/window-app.png" alt="下载的一些软件" >
</center>
