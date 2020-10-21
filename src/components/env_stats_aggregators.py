REGISTRY = {}

try:
    from envs.starcraft1 import StatsAggregator as SC1StatsAggregator
    REGISTRY["sc1"] = SC1StatsAggregator
except Exception as e:
    print(e)

try:
    from envs.starcraft2 import StatsAggregator as SC2StatsAggregator
    REGISTRY["sc2"] = SC2StatsAggregator
except Exception as e:
    print(e)
