from __future__ import unicode_literals

from django.db import models

# Create your models here.

class ArticleContent(models.Model): #TODO: decide what to make mandatory
    body = models.TextField()
    title = models.CharField(max_length=400, unique=False) # TODO: should we make this mandatory
    author = models.CharField(max_length=400, blank=True, null=True)


class ArchivedArticle(models.Model):
    created_at = models.DateField(auto_now_add=True)
    published_date = models.DateField(default=None) # TODO: should we make this necessarity
    wordpress_id = models.IntegerField(unique=True)    
    content = models.OneToOneField(ArticleContent, on_delete=models.CASCADE) # TODO: create delete bejvaior
    veridian_id = models.CharField(max_length=100, unique=True)


class CurrentArticle(models.Model):
    created_at = models.DateField(auto_now_add=True)
    published_date = models.DateField(default=None) # TODO: should we make this necessarity
    wordpress_id = models.IntegerField(unique=True)    
    content = models.OneToOneField(ArticleContent, on_delete=models.CASCADE) #TODO: create delete behavior
    archived_articles = models.ManyToManyField(ArchivedArticle)


    
