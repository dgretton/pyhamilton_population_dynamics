#!python3

import os, sys, logging
import numpy as np
from itertools import zip_longest

def neighbors_96(idx):
    return [i for i in (idx + 8, idx - 1, idx - 8, idx + 1) if 0 <= i < 96 and abs(idx%8 - i%8) <= 1]

def xy_96(idx):
    return idx//8, idx%8

mountain_peak = 8*8+4
def mountain_factor(idx):
    if idx == mountain_peak:
        return 0
    x, y = xy_96(idx)
    mx, my = xy_96(mountain_peak)
    return 1.0/(1.0 + 1.0/((x - mx)**2 + (y - my)**2))
    
flow_matrix = np.zeros((96, 96))

normal_rate = .16

for i in range(96):
    for n in neighbors_96(i):
        flow_matrix[i, n] = normal_rate

passability = np.zeros((96,)) + normal_rate

def apply_mountain(mountain_idxs, mountain_rate):
    for mountain_idx in mountain_idxs:
        passability[mountain_idx] = mountain_rate
    for i in range(96):
        for n in neighbors_96(i):
            if n in mountain_idxs or i in mountain_idxs:
                flow_matrix[i, n] = mountain_rate

apply_mountain([44+8, 44+9, 44+16, 44+17], normal_rate/20)
apply_mountain([44-9, 44-8], normal_rate/10)

with open('passability.csv', 'w+') as f:
    passability_grid = np.transpose(np.reshape(passability, (12, 8)))
    for row in passability_grid:
        f.write(','.join((str(n) for n in row)) + '\n')

#for mountain_i in mountain_idxs:
#    print(mountain_i)
#    for n in neighbors_96(mountain_i):
#        print('neighbor', n)
#        flow_matrix[mountain_i, n] /= 2

for i, column in enumerate(np.transpose(flow_matrix)):
    flow_matrix[i, i] = 1.0 - sum(column)

with open('flow_matrix.csv', 'w+') as f:
    for row in flow_matrix:
        f.write(','.join((str(n) for n in row)) + '\n')

start_vec = np.zeros((96,))
start_vec[44] = 1.0
#start_vec[8*10+4] = .3
#start_vec[8*6+2:8*6+7:2] = .2
#start_vec[8*2+4:8*8:8] = .2
num_steps = 12
states = []
for i in range(num_steps):
    states.append(np.transpose(np.reshape(start_vec, (12, 8))))
    start_vec = np.matmul(start_vec, flow_matrix)    

import matplotlib.pyplot as plt

if len(sys.argv) > 1 and sys.argv[1] == '--plot':
    plot_states = states[::1]
    plt.figure(figsize=(50, 10))
    for i, state in enumerate(plot_states):
        plt.subplot(2, len(plot_states)/2, i+1)
        plt.matshow(state, fignum=False, cmap=plt.cm.Reds, vmin=0, vmax=.1)
        #for i in range(15):
        #    for j in range(15):
        #        c = state[j,i]
        #        ax.text(i, j, str(c), va='center', ha='center')
    plt.savefig('gene_flow_simulation.pdf')
    plt.show()
    exit()


# ------------------------------------------------------------------------------------- #


this_file_dir = os.path.dirname(__file__)
methods_dir = os.path.abspath(os.path.join(this_file_dir, '..', '..'))

pyham_pkg_path =  os.path.join(methods_dir, 'perma_oem', 'pyhamilton')

if pyham_pkg_path not in sys.path:
    sys.path.append(pyham_pkg_path)

from pyhamilton import (HamiltonInterface, LayoutManager, ResourceType, Plate96, Tip96,
    INITIALIZE, PICKUP, EJECT, ASPIRATE, DISPENSE, PICKUP96, EJECT96, ASPIRATE96, DISPENSE96, oemerr)

def resource_list_with_prefix(layout_manager, prefix, res_class, num_ress):
    def name_from_line(line):
        field = LayoutManager.layline_objid(line)
        if field:
            return field
        return LayoutManager.layline_first_field(line)
    layline_test = lambda line: LayoutManager.field_starts_with(name_from_line(line), prefix)
    res_type = ResourceType(res_class, layline_test, name_from_line)
    res_list = [layout_manager.assign_unused_resource(res_type) for _ in range(num_ress)]
    res_list.sort(key=lambda r: r.layout_name())
    return res_list

def labware_pos_str(labware, idx):
    return labware.layout_name() + ', ' + labware.position_id(idx)

def compound_pos_str(pos_tuples):
    present_pos_tups = [pt for pt in pos_tuples if pt is not None]
    return ';'.join((labware_pos_str(labware, idx) for labware, idx in present_pos_tups))

def channel_var(pos_tuples):
    ch_var = ['0']*16
    for i, pos_tup in enumerate(pos_tuples):
        if pos_tup is not None:
            ch_var[i] = '1'
    return ''.join(ch_var)

# Tools for pipelining commands
wait_on_id = None

def block_on_last_command():
    global wait_on_id
    if wait_on_id is not None:
        ham_int.wait_on_response(wait_on_id, raise_first_exception=True)
    wait_on_id = None

def queue_id(cmd_id):
    global wait_on_id
    wait_on_id = cmd_id

def tip_pick_up(ham_int, pos_tuples):
    block_on_last_command()
    logging.info('tip_pick_up: Pick up tips at ' + '; '.join((labware_pos_str(*pt) if pt else '(skip)' for pt in pos_tuples)))
    num_channels = len(pos_tuples)
    if num_channels > 8:
        raise ValueError('Can only pick up 8 tips at a time')
    ch_patt = channel_var(pos_tuples)
    labware_poss = compound_pos_str(pos_tuples)
    # TODO: put in some error handling for absent tips etc.
    queue_id(ham_int.send_command(PICKUP, labwarePositions=labware_poss, channelVariable=ch_patt))

def tip_eject(ham_int, pos_tuples):
    block_on_last_command()
    logging.info('tip_eject: Eject tips to ' + '; '.join((labware_pos_str(*pt) if pt else '(skip)' for pt in pos_tuples)))
    num_channels = len(pos_tuples)
    if num_channels > 8:
        raise ValueError('Can only pick up 8 tips at a time')
    ch_patt = channel_var(pos_tuples)
    labware_poss = compound_pos_str(pos_tuples)
    queue_id(ham_int.send_command(EJECT, labwarePositions=labware_poss, channelVariable=ch_patt))

default_liq_class = 'HighVolumeFilter_Water_DispenseJet_Empty_with_transport_vol'

def assert_parallel_nones(list1, list2):
    if not (len(list1) == len(list2) and all([bool(i1) == bool(i2) for i1, i2 in zip(list1, list2)])):
        raise ValueError('Lists must have parallel None entries')

def aspirate(ham_int, pos_tuples, vols, **more_options):
    block_on_last_command()
    assert_parallel_nones(pos_tuples, vols)
    logging.info('aspirate: Aspirate volumes ' + str(vols) + ' from positions [' +
            '; '.join((labware_pos_str(*pt) if pt else '(skip)' for pt in pos_tuples)) +
            (']' if not more_options else '] with extra options ' + str(more_options)))
    if len(pos_tuples) > 8:
        raise ValueError('Can only aspirate with 8 channels at a time')
    queue_id(ham_int.send_command(ASPIRATE,
        channelVariable=channel_var(pos_tuples),
        labwarePositions=compound_pos_str(pos_tuples),
        volumes=[v for v in vols if v is not None],
        liquidClass=default_liq_class,
        **more_options))

def dispense(ham_int, pos_tuples, vols, **more_options):
    block_on_last_command()
    assert_parallel_nones(pos_tuples, vols)
    logging.info('dispense: Dispense volumes ' + str(vols) + ' into positions [' +
            '; '.join((labware_pos_str(*pt) if pt else '(skip)' for pt in pos_tuples)) +
            (']' if not more_options else '] with extra options ' + str(more_options)))
    if len(pos_tuples) > 8:
        raise ValueError('Can only aspirate with 8 channels at a time')
    queue_id(ham_int.send_command(DISPENSE,
        channelVariable=channel_var(pos_tuples),
        labwarePositions=compound_pos_str(pos_tuples),
        volumes=[v for v in vols if v is not None],
        liquidClass=default_liq_class,
        **more_options))

def tip_pick_up_96(ham_int, tips, **more_options):
    logging.info('tip_pick_up_96: Pick up tips from ' + tips.layout_name())
    pos_tuples = [(tips, i) for i in range(96)]
    labware_poss = compound_pos_str(pos_tuples)
    def go():
        ham_int.wait_on_response(ham_int.send_command(PICKUP96,
            labwarePositions=labware_poss,
            **more_options), raise_first_exception=True)
    try:
        go()
    except oemerr.TipPresentError:
        logging.info('Tips were already present, ejecting them first')
        tip_eject_96(ham_int, None, tipEjectToKnownPosition=2) # default waste! watch out
        go()

def tip_eject_96(ham_int, tips, **more_options):
    if tips:
        logging.info('tip_eject_96: Eject tips to ' + tips.layout_name())
        pos_tuples = [(tips, i) for i in range(96)]
        labware_poss = compound_pos_str(pos_tuples)
    else:
        logging.info('tip_eject_96, no tip location specified')
        labware_poss = ''
    ham_int.wait_on_response(ham_int.send_command(EJECT96,
        labwarePositions=labware_poss,
        **more_options), raise_first_exception=True)

def aspirate_96(ham_int, plate, vol, **more_options):
    logging.info('aspirate_96: Aspirate volume ' + str(vol) + ' from ' + plate.layout_name() +
            ('' if not more_options else ' with extra options ' + str(more_options)))
    pos_tuples = [(plate, i) for i in range(96)]
    ham_int.wait_on_response(ham_int.send_command(ASPIRATE96,
        labwarePositions=compound_pos_str(pos_tuples),
        aspirateVolume=vol,
        liquidClass=default_liq_class,
        **more_options), raise_first_exception=True)

def dispense_96(ham_int, plate, vol, **more_options):
    logging.info('dispense_96: Aspirate volume ' + str(vol) + ' from ' + plate.layout_name() +
            ('' if not more_options else ' with extra options ' + str(more_options)))
    pos_tuples = [(plate, i) for i in range(96)]
    ham_int.wait_on_response(ham_int.send_command(DISPENSE96,
        labwarePositions=compound_pos_str(pos_tuples),
        dispenseVolume=vol,
        liquidClass=default_liq_class,
        **more_options), raise_first_exception=True)

def move_vol_96(ham_int, plate1, plate2, vol):
    aspirate_96(ham_int, plate1, vol, airTransportRetractDist=10)
    dispense_96(ham_int, plate2, vol, airTransportRetractDist=10, dispenseMode=9, liquidHeight=10)

def wash(ham_int, tips, water_trough):
    tip_pick_up_96(ham_int, tips)
    aspirate_96(ham_int, water_trough, 800, mixVolume=800, mixCycles=2)
    dispense_96(ham_int, water_trough, 800, dispenseMode=9, liquidHeight=50)
    tip_eject_96(ham_int, tips)

def yield_in_chunks(sliceable, n):
    start_pos = 0
    end_pos = n
    while start_pos < len(sliceable):
        yield sliceable[start_pos:end_pos]
        start_pos, end_pos = end_pos, end_pos + n

def log_banner(banner_text):
    l = len(banner_text)
    margin = 5
    width = l + 2*margin + 2
    return ['#'*width,
            '#' + ' '*(width - 2) + '#',
            '#' + ' '*margin + banner_text + ' '*margin + '#',
            '#' + ' '*(width - 2) + '#',
            '#'*width]

log_dir = os.path.join(this_file_dir, 'log')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
main_logfile = os.path.join(log_dir, 'main.log')
total_move_vol = 250.0 # uL
well_vol = 300.0 # uL

layfile = os.path.join(this_file_dir, 'population_dynamics.lay')
lmgr = LayoutManager(layfile)

sys_state = lambda:None # simple namespace

plates = resource_list_with_prefix(lmgr, 'site_top_left_', Plate96, 10)
main_tips = lmgr.assign_unused_resource(ResourceType(Tip96, 'main_tips'))
wet_tips = lmgr.assign_unused_resource(ResourceType(Tip96, 'cross_tips'))
water_trough = lmgr.assign_unused_resource(ResourceType(Plate96, 'water_trough'))

def next_tips_tups():
    for column in range(11): # use prime number of columns to avoid sharing factors with other periodic activities
        yield [(main_tips, t) for t in range(8*column, 8*(column+1))]
tips_tups_gen = next_tips_tups()
tips_tups = None

wet_prep_plates = len(sys.argv) > 1 and sys.argv[1] == '--wet'

def system_initialize():
    ham_int, *_ = sys_state.instruments
    logging.basicConfig(filename=main_logfile, level=logging.DEBUG, format='[%(asctime)s] %(name)s %(levelname)s %(message)s')
    for banner_line in log_banner('Begin execution of ' + __file__):
        logging.info(banner_line)
    ham_int.set_log_dir(os.path.join(log_dir, 'hamilton.log'))
    ham_int.wait_on_response(ham_int.send_command(INITIALIZE), raise_first_exception=True)
    if wet_prep_plates:
        tip_pick_up_96(ham_int, wet_tips)
        for plate in plates[1:]:
            move_vol_96(ham_int, water_trough, plate, well_vol - total_move_vol)
        tip_eject_96(ham_int, wet_tips)
    wash(ham_int, main_tips, water_trough)

def groups_of_8(iter_in):
    return list(yield_in_chunks(list(iter_in), 8))

def nonzero_transfers(transfer_matrix, to_plate, for_matrix_rows):
    disp_agenda = [[] for _ in range(8)]
    disp_vols_agenda = [[] for _ in range(8)]
    for channel_idx, plate_idx in enumerate(for_matrix_rows):
        start_vec = np.zeros((96,))
        start_vec[plate_idx] = total_move_vol
        next_vec = np.matmul(start_vec, flow_matrix)
        for to_i, trans_vol in enumerate(next_vec):
            if trans_vol == 0:
                continue
            disp_agenda[channel_idx].append((to_plate, to_i)) # construct ragged array of transfer destinations for each channel
            disp_vols_agenda[channel_idx].append(trans_vol) # construct parallel ragged array of transfer volumes
    return zip_longest(*disp_agenda), zip_longest(*disp_vols_agenda) # fill in None entries to make ragged arrays square

def next_tips_wash_if_needed():
    tips_tups_gen = iter(())
    try:
        return next(tips_tups_gen)
    except StopIteration:
        wash(ham_int, main_tips, water_trough)
        tips_tups_gen = next_tips_tups() # reset generator
        return next(tips_tups_gen)

def load_new_tips():
    ham_int, *_ = sys_state.instruments
    sys_state.tips_tups = next_tips_wash_if_needed()
    tip_pick_up(ham_int, sys_state.tips_tups)

def put_tips_back():
    ham_int, *_ = sys_state.instruments
    tip_eject(ham_int, sys_state.tips_tups)

def parallel_source_positions_in(from_plate, column, transfer_destinations):
    return[None if dpt is None else (from_plate, c) for c, dpt in zip(column, transfer_destinations)]

def aspirate_from(positions, volumes):
    ham_int, *_ = sys_state.instruments
    aspirate(ham_int, positions, volumes)

def dispense_to(positions, volumes):
    ham_int, *_ = sys_state.instruments
    dispense(ham_int, positions, volumes, liquidHeight=9)

def main():
    for from_plate, to_plate in zip(plates, plates[1:]):
        for column in groups_of_8(range(96)):
            destinations_by_column, volumes_by_column = nonzero_transfers(flow_matrix, to_plate, column)
            for transfer_destinations, transfer_volumes in zip(destinations_by_column, volumes_by_column):
                load_new_tips()
                aspirate_from(parallel_source_positions_in(from_plate, column, transfer_destinations), transfer_volumes)
                dispense_to(transfer_destinations, transfer_volumes)
                put_tips_back()

class Nothing:
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

if __name__ == '__main__':
    with HamiltonInterface() as ham_int, \
            Nothing() as reader_int, \
            Nothing() as pump_int:
        sys_state.instruments = ham_int, reader_int, pump_int
        system_initialize()
        main()
