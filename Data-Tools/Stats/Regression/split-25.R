library(dplyr)

# responsibe for generating all figures for holdout approach

rm(list = ls())
setwd('~/Desktop/Repositories/lexidate-variation-analysis/Data-Tools/Stats/Regression/')
cat("\014")

NAMES = c('tournament', 'lexicase')
data_dir <- './'
r_task_id_lists <- c(359934, 359945, 359948, 359933)

# regression scores
regression_scores <- read.csv(paste(data_dir, '../holdout_data.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)
regression_scores$selection <- factor(regression_scores$selection, levels = NAMES)
regression_scores = filter(regression_scores, task_type == 'regression')

training_scores <- regression_scores[duplicated(regression_scores$seed) | duplicated(regression_scores$seed, fromLast = TRUE), ] %>%
  select(training_performance, selection, seed, task_id, split)
training_scores <- reshape(training_scores, timevar = 'selection', idvar = c('seed', 'task_id', 'split'), direction = "wide")
names(training_scores)[names(training_scores) == "training_performance.lexicase"] <- "lexicase"
names(training_scores)[names(training_scores) == "training_performance.tournament"] <- "tournament"
training_scores = filter(training_scores, split == '25%')

testing_scores <- regression_scores[duplicated(regression_scores$seed) | duplicated(regression_scores$seed, fromLast = TRUE), ] %>%
  select(testing_performance, selection, seed, task_id, split)
testing_scores <- reshape(testing_scores, timevar = 'selection', idvar = c('seed', 'task_id', 'split'), direction = "wide")
names(testing_scores)[names(testing_scores) == "testing_performance.lexicase"] <- "lexicase"
names(testing_scores)[names(testing_scores) == "testing_performance.tournament"] <- "tournament"
testing_scores = filter(testing_scores, split == '25%')

# stats for 359934
# training: t > l (0.0009928)
training = filter(training_scores, task_id == r_task_id_lists[1])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 'g', conf.int = FALSE)
# testing t = l (0.6742)
testing = filter(testing_scores, task_id == r_task_id_lists[1])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)

# stats for 359945
# training: t = l (0.2753)
training = filter(training_scores, task_id == r_task_id_lists[2])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)
# testing: t = l (0.6794)
testing = filter(testing_scores, task_id == r_task_id_lists[2])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)

# stats for 359948
# training: t = l (0.09551)
training = filter(training_scores, task_id == r_task_id_lists[3])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)
# testing: t = l (0.3525)
testing = filter(testing_scores, task_id == r_task_id_lists[3])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)

# stats for 359933
# training: t = l (0.5958)
training = filter(training_scores, task_id == r_task_id_lists[4])
wilcox.test(training$tournament, training$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)
# testing: t = l (0.165)
testing = filter(testing_scores, task_id == r_task_id_lists[4])
wilcox.test(testing$tournament, testing$lexicase, paired = TRUE, alternative = 't', conf.int = FALSE)