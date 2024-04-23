from gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus

GA3C_config = {
    11:{
        'Actions': Actions(),
        'medium':{
        'checkpt_dir': 'run-20240423_133923-dnpwgars/files/checkpoints',
        'checkpt_name': 'network_00200001'},
        'expert':{
        'checkpt_dir': 'run-20240423_110649-2xoksmf2/files/checkpoints',
        'checkpt_name': 'network_02000000'},
    },
    29:{
        'Actions': Actions_Plus(),
        'medium':{
        'checkpt_dir': 'run-20240422_115842-owleflbx/files/checkpoints',
        'checkpt_name': 'network_00200000'},
        # 'expert':{
        # 'checkpt_dir': 'run-20240322_114351-opn3hmir/files/checkpoints',
        # 'checkpt_name': 'network_00200000'},
    }
}