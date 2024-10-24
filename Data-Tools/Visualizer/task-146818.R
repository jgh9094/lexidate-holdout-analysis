# task 146818
task = 146818

# 10% split
test_10p = test_plot(task, '10%')
train_10p = train_plot(task, '10%')

# 20% split
test_20p = test_plot(task, '20%')
train_20p = train_plot(task, '20%')

# 50% split
test_50p = test_plot(task, '50%')
train_50p = train_plot(task, '50%')

# 80% split
test_80p = test_plot(task, '80%')
train_80p = train_plot(task, '80%')

# 90% split
test_90p = test_plot_last_col(task, '90%')
train_90p = train_plot_last_col(task, '90%')

# get legend
legend <- cowplot::get_legend(
  train_90p +
    guides(
      shape=guide_legend(nrow=1,title="Selection scheme"),
      color=guide_legend(nrow=1,title="Selection scheme"),
      fill=guide_legend(nrow=1,title="Selection scheme")
    ) +
    theme(
      legend.position = "top",
      legend.box="verticle",
      legend.justification="center"
    )
)

# make a single row plot
test_row <- test_row_p(test_10p, test_20p, test_50p, test_80p, test_90p)

train_row = train_row_p(train_10p, train_20p, train_50p, train_80p, train_90p)

plot = plot_grid(
  ggdraw() + draw_label("Accuracy on test set for OpenML task 146818", fontface='bold', size = 24) + p_theme,
  test_row,
  ggdraw() + draw_label("Accuracy on selection set for OpenML task 146818", fontface='bold', size = 24) + p_theme,
  train_row,
  legend,
  nrow = 5,
  rel_heights = c(0.12,1.0,0.12,1.0,0.06)
)

save_plot(
  paste(filename ="task-146818.pdf"),
  plot,
  base_width=10,
  base_height=10
)
