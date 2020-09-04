import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)

#adapted from https://docs.python.org/3/library/datetime.html#datetime.date.weekday
def week_number(date):
    return date.weekday()

def year_weeknumber(date):
    return date.isocalendar()[0], date.isocalendar()[1]

def main():
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
    #reddit_counts = "reddit-counts.json.gz"
    #counts = pd.read_json(reddit_counts, lines=True)
    # we will look only at values (1) in 2012 and 2013, and (2) in the /r/canada subreddit
    #adapted from https://www.w3resource.com/pandas/dataframe/dataframe-loc.php#:~:text=The%20loc()%20function%20is,used%20with%20a%20boolean%20array.
    counts_2012_canada = counts.loc[(counts['date'].dt.year==2012)& (counts['subreddit']=='canada')]
    counts_2013_canada = counts.loc[(counts['date'].dt.year==2013)& (counts['subreddit']=='canada')]
    counts_new = pd.concat([counts_2012_canada, counts_2013_canada]).reset_index(drop=True)

    counts_new['weeknumber'] = counts_new['date'].apply(week_number)
    counts_weekday = counts_new.loc[(counts_new['weeknumber']<5)].reset_index(drop=True)
    counts_weekend = counts_new.loc[(counts_new['weeknumber']>=5)].reset_index(drop=True)

    #print(counts_weekday)
    #print(counts_weekend)

    #Student's T-test
    t_test_p = stats.ttest_ind(counts_weekday['comment_count'], counts_weekend['comment_count']).pvalue
    print(f"Initial (invalid) T-test p-value: {t_test_p:.3g}\n")
    #print(t_test_p)
    weekday_normaltest  = stats.normaltest(counts_weekday['comment_count']).pvalue
    weekend_normaltest  = stats.normaltest(counts_weekend['comment_count']).pvalue
    #print(weekday_normaltest) 1.0091137251707687e-07 -> not normal
    #print(weekend_normaltest) 0.0015209196859635404 -> not normal
    levene_test = stats.levene(counts_weekday['comment_count'], counts_weekend['comment_count']).pvalue
    print(f"Original data normality p-values: {weekday_normaltest:.3g} {weekend_normaltest:.3g}\n")
    print(f"Original data equal-variance p-value: {levene_test:.3g}\n")
    #print(levene_test)     0.04378740989202803 -> variances are different

    #=========Fix 1
    # plt.hist(counts_weekday['comment_count'])
    # plt.hist(counts_weekend['comment_count'])
    # plt.show()
    weekday_log = np.log(counts_weekday['comment_count'])
    weekend_log = np.log(counts_weekend['comment_count'])
    # plt.hist(weekday_log)
    # plt.hist(weekend_log)
    # plt.show()
    weekday_log_p = stats.normaltest(weekday_log).pvalue #0.0004015914200681448 -> not normal
    weekend_log_p = stats.normaltest(weekend_log).pvalue #0.3149388682066699 -> normal
    log_levene = stats.levene(weekday_log, weekend_log).pvalue  #0.0004190759393372205 -> variances are different
    # print(f"Transformed data normality p-values: {weekday_log_p:.3g} {weekend_log_p:.3g}\n")
    # print(f"Transformed data equal-variance p-value: {log_levene:.3g}\n")


    weekday_sqrt = np.sqrt(counts_weekday['comment_count'])
    weekend_sqrt = np.sqrt(counts_weekend['comment_count'])
    # plt.hist(weekday_sqrt)
    # plt.hist(weekend_sqrt)
    # plt.show()
    weekday_sqrt_p = stats.normaltest(weekday_sqrt).pvalue #0.0004015914200681448 -> not normal
    weekend_sqrt_p = stats.normaltest(weekend_sqrt).pvalue #0.3149388682066699 -> normal
    log_levene_sqrt = stats.levene(weekday_sqrt, weekend_sqrt).pvalue  #0.0004190759393372205 -> variances are different
    print(f"Transformed data normality p-values: {weekday_sqrt_p:.3g} {weekend_sqrt_p:.3g}\n")
    print(f"Transformed data equal-variance p-value: {log_levene_sqrt:.3g}\n")

    #=========Fix 2

    #https://docs.python.org/3//library/datetime.html#datetime.date.isocalendar
    counts_weekday['year_week'] = counts_weekday['date'].apply(year_weeknumber)
    counts_weekend['year_week'] = counts_weekend['date'].apply(year_weeknumber)

    counts_weekday_f2 = counts_weekday.groupby(['year_week']).mean()
    counts_weekend_f2 = counts_weekend.groupby(['year_week']).mean()

    counts_weekday_f2_p = stats.normaltest(counts_weekday_f2['comment_count']).pvalue # 0.3082637390825463 -> normal distribution
    counts_weekend_f2_p = stats.normaltest(counts_weekend_f2['comment_count']).pvalue # 0.15294924717078442 -> normal distriution
    f2_ttest_p = stats.ttest_ind(counts_weekday_f2['comment_count'], counts_weekend_f2['comment_count'])[1] #pvalue=1.3353656052303144e-34
    f2_levene_p = stats.levene(counts_weekday_f2['comment_count'], counts_weekend_f2['comment_count'])[1]  #pvalue=0.20383788083573426 -> vaariances are same

    print(f"Weekly data normality p-values: {counts_weekday_f2_p:.3g} {counts_weekend_f2_p:.3g}\n")
    print(f"Weekly data equal-variance p-value: {f2_levene_p:.3g}\n")
    print(f"Weekly T-test p-value: {f2_ttest_p:.3g}\n")
    #=========Fix 3
    #adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    mann_utest = stats.mannwhitneyu(counts_weekday['comment_count'], counts_weekend['comment_count'], alternative = "two-sided")[1]
    print(f"Mann–Whitney U-test p-value: {mann_utest:.3g}")

# ...

print(OUTPUT_TEMPLATE.format(
    initial_ttest_p=0,
    initial_weekday_normality_p=0,
    initial_weekend_normality_p=0,
    initial_levene_p=0,
    transformed_weekday_normality_p=0,
    transformed_weekend_normality_p=0,
    transformed_levene_p=0,
    weekly_weekday_normality_p=0,
    weekly_weekend_normality_p=0,
    weekly_levene_p=0,
    weekly_ttest_p=0,
    utest_p=0,
))


if __name__ == '__main__':
    main()
