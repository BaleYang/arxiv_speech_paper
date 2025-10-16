---
layout: default
title: Latest
---

{% assign latest_post = site.posts | first %}

{% if latest_post %}
<article class="post h-entry" itemscope itemtype="http://schema.org/BlogPosting">
  <header class="post-header">
    <h1 class="post-title p-name" itemprop="name headline">{{ latest_post.title }}</h1>
    <p class="post-meta">
      <time class="dt-published" datetime="{{ latest_post.date | date_to_xmlschema }}" itemprop="datePublished">
        {{ latest_post.date | date: "%Y-%m-%d" }}
      </time>
    </p>
  </header>

  <div class="post-content e-content" itemprop="articleBody">
    {{ latest_post.content }}
  </div>
</article>
{% else %}
<p>暂无文章。</p>
{% endif %}

