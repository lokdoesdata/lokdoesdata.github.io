---
layout: splash
classes:
  - landing
permalink: /
header:
  overlay_color: '#5e616c'
  overlay_image:  /assets/images/home-unsplash.jpg
  cta_label: 
  cta_url: 
  caption: "Photo credit: [<u>NASA</u>](https://unsplash.com/photos/Q1p7bh3SHj8) on [<u>Unsplash</u>](https://unsplash.com/)"
excerpt: 'Portfolio of work by Lok Ngan.<br /><br /> [<i class="fab fa-linkedin"></i>](https://www.linkedin.com/in/lokngan/){: .btn .btn--primary .btn--small} &nbsp; [<i class="fab fa-github"></i>](https://github.com/lokdoesdata){: .btn .btn--primary .btn--small} &nbsp; [<i class="fab fa-medium"></i>](https://medium.com/@lokdoesdata){: .btn .btn--primary .btn--small} &nbsp; [<i class="fab fa-reddit"></i>](https://www.reddit.com/user/lokdoesdata/){: .btn .btn--primary .btn--small}'

data_row:
  - image_path: /assets/images/portfolio/income-inequality/us-income-inequality-thumbnail.png
    alt: "data"
    title: "Data Collection"
    excerpt: 'Combining data and generating information from multiple sources.'
    url: "/portfolio"
    btn_label: "Read More"
    btn_class: "btn--inverse"
analytic_row:
  - image_path: /assets/images/portfolio/fashion-mnist/tsne-thumbnail.png
    alt: "analysis"
    title: 'Data Analytics'
    excerpt: "Analysis based on traditional statisical analysis, machine learning, and deep learning."
    url: "/portfolio"
    btn_label: "Read More"
    btn_class: "btn--inverse"
visual_row:
  - image_path: /assets/images/portfolio/billion-dollar-climate/billion-dollar-climate-thumbnail.png
    alt: "visual"
    title: "Information Visualizations"
    excerpt: "Convey information through visualizations."
    url: "/portfolio"
    btn_label: "Read More"
    btn_class: "btn--inverse"
---

{% include feature_row  id='data_row' type='left' %}

{% include feature_row  id='analytic_row' type='right' %}

{% include feature_row  id='visual_row' type='left' %}
