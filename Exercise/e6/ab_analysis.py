import sys
import numpy as np
import pandas as pd
import scipy.stats as stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    #searchdata_file = "searches.json"
    search_file = pd.read_json(searchdata_file, orient='records', lines=True)
    #print(search_file)
    #odd is new interface
    odd = search_file[search_file["uid"]%2==1].reset_index(drop=True)
    even = search_file[search_file["uid"]%2==0].reset_index(drop=True)

    odd_searches = odd[odd["search_count"]>0]
    even_searches = even[even["search_count"]>0]

    #print(odd_searches)
    odd_searches_number = odd_searches.shape[0] #number of rows that search at least once
    odd_searches_zero = (odd.shape[0] - odd_searches.shape[0])  #number of rows that never search

    even_searches_number = even_searches.shape[0] #number of rows
    even_searches_zero = (even.shape[0] - even_searches.shape[0])

    #adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    obs = np.array([[odd_searches_number, odd_searches_zero], [even_searches_number, even_searches_zero]])
    more_users_p  =stats.chi2_contingency(obs)[1]
    #print(more_users_p)

    ###################################
    #repeat again with instructors only
    inst_odd = odd[odd["is_instructor"]==True]
    inst_even = even[even["is_instructor"]==True]
    inst_odd_searches = inst_odd[inst_odd["search_count"]>0]
    inst_even_searches = inst_even[inst_even["search_count"]>0]

    inst_odd_searches_number = inst_odd_searches.shape[0] #number of rows that search at least once
    inst_odd_searches_zero = (inst_odd.shape[0] - inst_odd_searches.shape[0])  #number of rows that never search

    inst_even_searches_number = inst_even_searches.shape[0] #number of rows
    inst_even_searches_zero = (inst_even.shape[0] - inst_even_searches.shape[0])

    inst_obs = np.array([[inst_odd_searches_number, inst_odd_searches_zero], [inst_even_searches_number, inst_even_searches_zero]])
    more_instr_p =stats.chi2_contingency(inst_obs)[1]

    #more_searches_p
    more_searches_p = stats.mannwhitneyu(odd["search_count"], even["search_count"]).pvalue
    more_instr_searches_p = stats.mannwhitneyu(inst_odd["search_count"], inst_even["search_count"]).pvalue
    # ...

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p= more_users_p,
        more_searches_p=more_searches_p,
        more_instr_p=more_instr_p,
        more_instr_searches_p=more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
