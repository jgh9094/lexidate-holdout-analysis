# responsibe for generating all figures for holdout approach

rm(list = ls())

library(dplyr)
library(dplyr)
library(ggplot2)

all_elements_same <- function(lst) {
  # Use all.equal to compare elements, considering floating-point precision
  identical_value <- lst[[1]]
  all(sapply(lst, function(x) all.equal(x, identical_value)) == TRUE)
}

permutation_test <- function(x, y, seed = 123, alternative = "t", n_permutations = 500000) {
  # Set the random seed for reproducibility
  set.seed(seed)
  
  if(all_elements_same(x) & all_elements_same(y))
  {
    print('P-value: Both columns are equal')
    return(-1)
  }
  
  # Calculate the observed difference in means
  if (alternative == "t") {
    # observed_diff <- abs(mean(x) - mean(y))
    observed_diff <- abs(t.test(x, y, var.equal = FALSE)$statistic)
  }
  else{
    observed_diff <- t.test(x, y, var.equal = FALSE)$statistic
  }
  print('observed difference')
  print(observed_diff)
  
  # Combine both samples
  combined <- c(x, y)
  n_x <- length(x)
  
  # Generate permutation differences
  permutation_diffs <- numeric(n_permutations)
  
  # Use a reproducible random sequence for each permutation
  seeds <- sample.int(1e8, n_permutations)  # Generate unique seeds for each permutation
  
  for (i in 1:n_permutations) {
    set.seed(seeds[i])  # Set seed for this permutation
    permuted <- sample(combined)  # Shuffle the combined data
    perm_x <- permuted[1:n_x]     # First n_x elements to group 1
    perm_y <- permuted[(n_x + 1):length(combined)]  # Remaining elements to group 2
    
    # permutation_diffs[i] <- mean(perm_x) - mean(perm_y)
    permutation_diffs[i] <- t.test(perm_x, perm_y, var.equal = FALSE)$statistic
    # print(permutation_diffs[i])
  }
  
  # Compute the p-value based on the specified alternative
  if (alternative == "l") {
    p_value <- mean(permutation_diffs <= observed_diff)
  } else if (alternative == "g") {
    p_value <- mean(permutation_diffs >= observed_diff)
  } else if (alternative == "t") {
    p_value <- mean(abs(permutation_diffs) >= abs(observed_diff))
  } else {
    stop("Invalid alternative. Choose from 'less', 'greater', or 'two-sided'.")
  }
  
  # Create the histogram with ggplot2
  if (alternative == "t") {
    df <- data.frame(Difference = abs(permutation_diffs))
  }
  else{
    df <- data.frame(Difference = permutation_diffs)
  }
  
  plot <- ggplot(df, aes(x = Difference)) +
    geom_histogram(
      aes(y = after_stat(density)),
      bins = 50, 
      fill = "lightgreen", 
      color = "white", 
      alpha = 0.7
    ) +
    geom_vline(
      xintercept = observed_diff, 
      color = "black", 
      linetype = "dashed", 
      size = 1
    ) +
    labs(
      title = "Permutation Test: Distribution of  Mean Differences",
      x = "Difference in means",
      y = "Frequency"
    ) +
    theme_minimal()
  
  print(plot)  # Display the plot
  
  # Return the p-value
  print('p-value')
  print(p_value)
}

setwd('~/Desktop/Repositories/lexidate-variation-analysis/Data-Tools/Stats/')
cat("\014")
c_task_id_lists <- c(146818,359954,359955,190146,168757,359956,359958,359959,2073,359960,168784,359962)
data_dir <- './'
NAMES = c('tournament', 'lexicase')


# classification scores
classification_scores <- read.csv(paste(data_dir, '../data.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)
classification_scores$selection <- factor(classification_scores$selection, levels = NAMES)
classification_scores <- filter(classification_scores, task_type == 'classification')

# Load necessary libraries
library(dplyr)

# List of task_ids
c_task_id_lists <- c(146818, 359954, 359955, 190146, 168757,
                     359956, 359958, 359959, 2073, 359960,
                     168784, 359962)

# Define the function to summarize and run statistical tests
summarize_and_test <- function(spliter, rng_seed1, rng_seed2) {
  for (id in c_task_id_lists) {
    cat("######### Task ID:", id, "#########\n")
    
    data <- filter(classification_scores, task_id == id & split == spliter)
    data$selection <- factor(data$selection, levels = NAMES)
    tournament <- filter(data, selection == 'tournament')
    lexicase <- filter(data, selection == 'lexicase')
    
    # Filter and summarize data for the current task_id
    cat("\n#### Testing", "####\n")
    summary_stats_test <- data %>%
                        group_by(selection) %>%
                        dplyr::summarise(
                          count = n(),
                          na_cnt = sum(is.na(testing_performance)),
                          min = min(testing_performance, na.rm = TRUE),
                          median = median(testing_performance, na.rm = TRUE),
                          mean = mean(testing_performance, na.rm = TRUE),
                          max = max(testing_performance, na.rm = TRUE),
                          IQR = IQR(testing_performance, na.rm = TRUE)
                        )
    # Print summary statistics
    print(summary_stats_test)
    
    
    # Run the Wilcoxon test
    permutation_test(tournament$testing_performance, lexicase$testing_performance, alternative = 'g', seed = rng_seed1)

    cat("\n#### Training", "####\n")
    summary_stats_train <- data %>%
                      group_by(selection) %>%
                      dplyr::summarise(
                        count = n(),
                        na_cnt = sum(is.na(training_performance)),
                        min = min(training_performance, na.rm = TRUE),
                        median = median(training_performance, na.rm = TRUE),
                        mean = mean(training_performance, na.rm = TRUE),
                        max = max(training_performance, na.rm = TRUE),
                        IQR = IQR(training_performance, na.rm = TRUE)
                      )
    # Print summary statistics
    print(summary_stats_train)

    # Run the Wilcoxon test
    permutation_test(tournament$training_performance, lexicase$training_performance, alternative = 'l', seed = rng_seed2)
    
    cat("\n###################################################################################################\n")
    
  }
}

# split 10 stats
output_file <- "stats-sum-10.txt"
sink(output_file)
summarize_and_test('10%', 0, 1)
sink()

# split 20 stats
output_file <- "stats-sum-20.txt"
sink(output_file)
summarize_and_test('20%',2,3)
sink()

# split 50 stats
output_file <- "stats-sum-50.txt"
sink(output_file)
summarize_and_test('50%',4,5)
sink()

# split 80 stats
output_file <- "stats-sum-80.txt"
sink(output_file)
summarize_and_test('80%',6,7)
sink()

# split 90 stats
output_file <- "stats-sum-90.txt"
sink(output_file)
summarize_and_test('90%',8,9)
sink()