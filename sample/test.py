import pandas as pd
import numpy as np
from inventorium.ri import switch_keys, capacity


class Inventory:
    def __init__(
            self,
            material,
            category,
            regional_dict,
            region_to_dc,
            pre_allocation_dict,
            first_dc_dict,
            dormant_dc,
            regions,
            index,
            dataset,
            n_dcs):
        self.material = material
        self.category = category
        self.regional_dict = regional_dict
        self.region_to_dc = region_to_dc
        self.pre_allocation_dict = pre_allocation_dict
        self.first_dc_dict = first_dc_dict
        self.dormant_dc = dormant_dc
        self.regions = regions
        self.n_dcs = n_dcs
        self.dataset = dataset
        self.index = index

    def select_top_regions(self):
        regions = self.regions
        index = self.index
        n_dcs = self.n_dcs + 2
        dataset = self.dataset

        regional_percents = dataset[regions].iloc[index].to_dict()

        reg_values = pd.DataFrame(regional_percents, [0]).T
        reg_values.columns = ['percent']

        # where we have ties we need to break them
        reg_values = reg_values.sort_values('percent', ascending=False)
        tie_breakers = [.04, .03, .02, .01]
        reg_values = reg_values.head(n_dcs)

        i = 0
        for index, row in reg_values.iterrows():
            reg_values.loc[index, ('percent')] = tie_breakers[i] + reg_values.loc[index, ('percent')]
            i += 1

        top_regions = {}
        for index in range(1, n_dcs+1):
            top_regions[index] = reg_values.apply(lambda x: x.nlargest(index).idxmin()).iloc[0]
        # end breaking ties
        
        self.top_regions = top_regions

        return self

    def assign_region_to_dc(self,exclusion,replicated_item):

        top_regions = self.top_regions
        regional_dict = self.regional_dict
        region_to_dc = self.region_to_dc

        region_copy = top_regions.copy()
        count = 1

        # We run a while loop to avoid having the same DC assigned. This is due to logic
        # where we have to prohibit a replicated item from ending up at a given DC per the stakeholder.
        while count < 100:
            for region_index, region_name in top_regions.items():

                regional_dict, dc = switch_keys(region_to_dc, region_name)
                if dc == exclusion and replicated_item == True:
                    top_regions[region_index] = {'region': 'Remove'}
                else:
                    top_regions[region_index] = {'region': region_name, 'dc': dc}

            # copy dictionary to avoid RuntimeError: dictionary changed size during iteration
            copy_d = top_regions.copy()

            for key in copy_d:
                if copy_d[key]['region'] == 'Remove':
                    top_regions.pop(key)

            # the system was setup where not all regions had back and it is trying to assign the same DC to handle two regions at once
            if count > 15:
                keys_to_keep = [1,3]
                top_regions = {key: top_regions[key] for key in keys_to_keep if key in top_regions}
            if count > 20:
                keys_to_keep = [1,4]
                top_regions = {key: top_regions[key] for key in keys_to_keep if key in top_regions}
        
            # reindex
            dict_new = {i + 1: v for i, (k, v) in enumerate(top_regions.items())}

            # return the first two
            dict_new = {k: dict_new[k] for k in list(dict_new.keys())[:2]}

            try:
                if dict_new[1]['dc'] != dict_new[2]['dc']:
                    # this will break us out of the while loop
                    count = 99
                else:
                    top_regions = region_copy.copy()
            except:
                top_regions = region_copy.copy()
            count = count + 1

        self.regional_dict = regional_dict
        self.top_regions = dict_new
        self.count = count

        return self

    def keep_dcs_unique(self):
        pass

    def category_exclusion(self):
        top_regions = self.top_regions
        region_to_dc = self.region_to_dc
        dormant_dc = self.dormant_dc

        if dormant_dc != '':
            li = []
            for key in top_regions.keys():
                current_dc = top_regions[key]['dc']
                if current_dc == dormant_dc:
                    li.append(1)
                    top_regions[key]['change_dc'] = True
                else:
                    top_regions[key]['change_dc'] = False
                    li.append(0)

            # we might not even need to change the dc
            if sum(li) > 0:
                tmp = pd.DataFrame(region_to_dc).fillna(0).T
                tmp['region'] = tmp.index
                tmp = tmp.reset_index(drop=True)
                tmp = pd.melt(tmp, id_vars=["region"]).query("value > 0")
                tmp['status'] = np.where(tmp['variable'] == dormant_dc, 'closed', 'open')

                # Change the DC
                for key in top_regions.keys():
                    change_dc = top_regions[key]['change_dc']
                    region = top_regions[key]['region']
                    if change_dc:
                        value = tmp.query(f"region == '{region}' and status == 'open'")['variable']
                        if len(value) > 0:
                            value = value.iloc[0]
                        else:
                            left_over_df = len(tmp.query("status == 'open'"))
                            if left_over_df > 0:
                                value = tmp.query("status == 'open'")['variable'].iloc[0]
                            else:
                                value = 'No DC available'

                        top_regions[key]['dc'] = value

            self.top_regions = top_regions

            return self

    def check_preallocation(self):

        material = self.material
        pre_allocation_dict = self.pre_allocation_dict
        if material in pre_allocation_dict.keys():
            self.dc_allocation = pre_allocation_dict[material]
        else:
            self.dc_allocation = ''

        return self

    def move_preallocation(self):

        dc_allocation = self.dc_allocation
        top_regions = self.top_regions
        first_dc_dict = self.first_dc_dict

        if len(dc_allocation) > 0:

            # first check for matches
            for key, value in top_regions.items():
                dst_cntr = top_regions[key]['dc']
                if dst_cntr in dc_allocation:
                    idx = dc_allocation.index(dst_cntr)
                    top_regions[key]['verified'] = True
                    dc_allocation.pop(idx)
                    top_regions[key]['method'] = 'No need to update'
                else:
                    top_regions[key]['verified'] = False

            # move remainder if necessary that match on region
            for pre_selected_dc in dc_allocation:
                tmp = pd.DataFrame(top_regions).T
                region = first_dc_dict[pre_selected_dc]
                tmp = tmp.query(f"region == '{region}' and verified != True")
                if len(tmp) == 1:
                    key_index = list(tmp.index)[0]
                    top_regions[key_index]['verified'] = True
                    top_regions[key_index]['dc'] = pre_selected_dc
                    top_regions[key_index]['method'] = 'Region Match'

                    # remove distribution center
                    idx = dc_allocation.index(pre_selected_dc)
                    dc_allocation.pop(idx)

            # move any remaining values
            if len(dc_allocation) > 0:
                for key, value in top_regions.items():
                    if not top_regions[key]['verified']:
                        top_regions[key]['dc'] = dc_allocation[0]
                        top_regions[key]['verified'] = True
                        top_regions[key]['method'] = 'Last Attempt'
                        dc_allocation.pop(0)
                        if len(dc_allocation) == 0:
                            break
            self.top_regions = top_regions

            return self

    def check_capacity(self, dataset, dc_capacity):

        region_to_dc = self.region_to_dc
        index = self.index

        # Check capacity
        if index > 1:
            capacity_output = capacity(dataset, dc_capacity)
            for row_index, row_capacity in capacity_output.iterrows():
                dst_cntr = row_capacity['dc']
                for key, value in region_to_dc.items():
                    try:
                        region_to_dc[key].pop(dst_cntr)
                    except BaseException:
                        pass

            self.capacity_output = capacity_output
            self.region_to_dc = region_to_dc

        return self
