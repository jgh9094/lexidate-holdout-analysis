# responsibe for generating all figures for holdout approach

rm(list = ls())
setwd('~/Desktop/Repositories/lexidate-variation-analysis/Data-Tools/Stats/Classification/')
cat("\014")

NAMES = c('tournament', 'lexicase')
data_dir <- './'
c_task_id_lists <- c(146818, 168784, 190137, 359969)

# classification scores
classification_scores <- read.csv(paste(data_dir, '../holdout_data.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)
classification_scores$selection <- factor(classification_scores$selection, levels = NAMES)
classification_scores <- filter(classification_scores, task_type == 'classification')

training_scores <- classification_scores[duplicated(classification_scores$seed) | duplicated(classification_scores$seed, fromLast = TRUE), ] %>%
  select(training_performance, selection, seed, task_id, split)
training_scores <- reshape(training_scores, timevar = 'selection', idvar = c('seed', 'task_id', 'split'), direction = "wide")
names(training_scores)[names(training_scores) == "training_performance.lexicase"] <- "lexicase"
names(training_scores)[names(training_scores) == "training_performance.tournament"] <- "tournament"
training_scores = filter(training_scores, split == '25%')

testing_scores <- classification_scores[duplicated(classification_scores$seed) | duplicated(classification_scores$seed, fromLast = TRUE), ] %>%
  select(testing_performance, selection, seed, task_id, split)
testing_scores <- reshape(testing_scores, timevar = 'selection', idvar = c('seed', 'task_id', 'split'), direction = "wide")
names(testing_scores)[names(testing_scores) == "testing_performance.lexicase"] <- "lexicase"
names(testing_scores)[names(testing_scores) == "testing_performance.tournament"] <- "tournament"
testing_scores = filter(testing_scores, split == '25%')

# stats for 146818
# training: t < l (0.0002292)
training = filter(training_scores, task_id == c_task_id_lists[1])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 'l', conf.int = FALSE)
# testing t > l (0.01021)
testing = filter(testing_scores, task_id == c_task_id_lists[1])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 'g', conf.int = FALSE)

# stats for 168784
# training: t < l (0.02087)
training = filter(training_scores, task_id == c_task_id_lists[2])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 'l', conf.int = FALSE)
# testing: t > l (0.04073)
testing = filter(testing_scores, task_id == c_task_id_lists[2])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 'g', conf.int = FALSE)

# stats for 190137
# training: t < l (0.0005057)
training = filter(training_scores, task_id == c_task_id_lists[3])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 'l', conf.int = FALSE)
# testing: t = l (1)
testing = filter(testing_scores, task_id == c_task_id_lists[3])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)

# stats for 359969
# training: t > l (4.778e-05)
training = filter(training_scores, task_id == c_task_id_lists[4])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 'g', conf.int = FALSE)
# testing: t = l (0.9198)
testing = filter(testing_scores, task_id == c_task_id_lists[4])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)