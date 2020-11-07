# 👾 Anime Match 👾
----
A multi-label classification model that uses transcipts from Anime to filter out specific themes such as Death, Romance, and Supernatural. 

## 👾 The Problem 👾
---
What do we consider when choosing anime? Genre, art style, content length, and popularity are typically what we think about.

Current recommendation systems that only filter by broad genre tags allow for specific themes to slip through and also close of entire categories for general thematic elements that are assumed. Broad genres, including horror and fantasy, may include several anime that deal with death and supernatural elements. Any individual that wished to not see either of these categories may eliminate the broad genre, where several shows in that genre don’t deal with either theme. 

Alternatively, there are several instances where an anime may be tagged by a traditionally light- hearted genre but do include darker themes. For example, the anime “Your Lie in April” is included in the romance genre overall but regularly includes themes of death from the main character’s relative passing to one of the main characters passing by the finale. By focusing on a thematic filter, we can add to existing recommendation systems to better the experience of anime enthusiasts, both by reducing the amount of anime with minor mentions of exclusionary themes to slip through due to their overarching genre tags, and by broadening the available recommendations with previously excluded larger genres.

## 👾 The Big Picture 👾
----
Imagine the last time you watched a movie in theatres (post-COV19, of course), what was it like coming across that uncomfortable scene you weren't prepared for? Wouldn't it be nice to have some kind of warning at the top of the screen so you wouldn't have to deal with elements of rape, murder, or kidnapping? Or maybe you're watching the newest episode of Rick and Morty and are not in the mood to watch a musical that day- wouldn't it be great to look at the description, see it tagged "musical", and skip it? Although this classficiation model focuses on Anime, it can also apply to Manga, TV shows, books, newspapers, and even parental control on your child's device.


## 👾 The Dataset 👾
---
The datasets uses in this project are raw transcripts from [Kistunekko](https://kitsunekko.net). It contains transcripts from over 2000 anime and in 4 languages (English, Japanese, Chinese, and Korean). For the purposes of this project, we will be sticking with English.

Transcipts can be found in the [/content/drive/My Drive/Colab Notebooks/transcripts]() path. Or you can download it usiing the source code from my [kistunekko-scrub](https://github.com/Imanisima/kitsunekko-scrub) repo!
