from gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus

GA3C_config = {
    11:{
        'Actions': Actions(),
        'medium':{
        'checkpt_dir': 'run-20240423_133923-dnpwgars',
        'checkpt_name': 'network_00200001'},
        'expert':{
        'checkpt_dir': 'run-20240423_235312-1cye0npr',
        'checkpt_name': 'network_01990000'},
    },
    29:{
        'Actions': Actions_Plus(),
        'medium':{
        'checkpt_dir': 'run-20240422_115842-owleflbx',
        'checkpt_name': 'network_00200000'},
        'expert':{
        'checkpt_dir': 'run-20240424_153727-928y04dp',
        'checkpt_name': 'network_01990000'},
    }
}