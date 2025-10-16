---
layout: page
title: Previous posts
permalink: /previous/
---

下面列出历史文章：

<ul class="post-list">
{% for post in site.posts %}
  <li>
    <span class="post-meta">{{ post.date | date: "%Y-%m-%d" }}</span>
    <h3>
      <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h3>
  </li>
{% endfor %}
</ul>

<p><a href="/">返回最新</a></p>

