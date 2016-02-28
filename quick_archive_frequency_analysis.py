# Quick Script to Calculate the frequency distribution for recommendations

from archive_app.similarity.models import *
import collections

# Get list of all archived articles and set to 0
archive_map = collections.Counter()
for archive in ArchivedArticle.objects.all():
    archive_map[archive.id] = 0

# Find count for each recommendation
for current in CurrentArticle.objects.all():
    for archive in current.archived_articles.all():
        archive_map[archive.id] += 1

print archive_map.most_common()

# Find how many with zero
zero_c = 0
for i in archive_map:
    if archive_map[i] == 0:
        zero_c += 1


# Find out total sum of recommendations
total_sum = 0
for a in archive_map:
    total_sum += archive_map[a]

# Avg # of recs per current article
total_recs = 0
denom = 0
for current in CurrentArticle.objects.all():
    total_recs += current.archived_articles.all().count()
    denom += 1

avg_rec = total_recs / float(denom)

