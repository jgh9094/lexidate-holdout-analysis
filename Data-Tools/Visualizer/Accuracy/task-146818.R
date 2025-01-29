# task 146818
task = 146818

# 5% split
test_5p = test_plot(task, '5%')

# 10% split
test_10p = test_plot(task, '10%')

# 50% split
test_50p = test_plot(task, '50%')

# 90% split
test_90p = test_plot_last_col(task, '90%')

# 95% split
test_95p = test_plot_last_col(task, '95%')

# get legend
legend <- cowplot::get_legend(
  train_95p +
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
test_row <- test_row_p(test_5p, test_10p, test_50p, test_90p, test_95p)

plot = plot_grid(
  ggdraw() + draw_label("Accuracy on test set for OpenML task 2073", fontface='bold', size = 24) + p_theme,
  test_row,
  legend,
  nrow = 5,
  rel_heights = c(0.12,1.0,0.12,1.0,0.06)
)

save_plot(
  paste(filename ="task-2073.pdf"),
  plot,
  base_width=10,
  base_height=10
)