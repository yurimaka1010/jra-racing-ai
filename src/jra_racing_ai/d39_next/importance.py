"""Bilingual saved-tree importance only; no inference, records, DB, or fitting."""
from ..ml import safety
from ..model_diagnostics import importance as saved_importance
from .contracts import EXTRA

JAPANESE={
 'venue':'競馬場','surface':'コース種別コード','going':'馬場状態','sex':'性別',
 'distance':'距離（m）','declared_runners':'登録頭数','frame':'枠番','horse_number':'馬番',
 'age':'馬齢','weight_kg':'負担重量（kg）','past_starts':'有効な過去出走数',
 'days_since_previous':'前回有効走からの日数','previous_finish':'前回有効走の着順',
 'previous_time_seconds':'同距離・同種別の有効過去タイム（秒）',
 'past_mean_finish':'過去平均着順','same_distance_mean_finish':'同距離の過去平均着順',
 'same_course_mean_finish':'同コースの過去平均着順',
 'a_previous_finish_percentile':'前走着順（頭数正規化）',
 'a_recent3_finish_percentile':'近3走着順（近走重視・頭数正規化）',
 'b_grade':'現在レースのグレード区分','b_condition':'現在レースの競走条件',
 'b_race_type':'競走種別','b_weight_type':'負担重量区分',
 'b_previous_grade':'前走グレード区分','b_previous_condition':'前走競走条件',
 'b_condition_transition':'競走条件の変化','b_grade_transition':'グレード区分の変化',
 'c_previous_late3f_seconds':'前走上がり3F（秒）',
 'c_previous_late3f_rank_percentile':'前走上がり順位（頭数正規化）',
 'c_previous_late3f_minus_race_mean':'前走上がり3Fのレース平均との差',
 'c_recent3_late3f_rank_percentile':'近3走上がり順位（近走重視・正規化）',
 'c_previous_corner4_percentile':'前走4角通過順位（頭数正規化）',
 'c_previous_adjacent_margin_lengths':'前走の隣接着差（馬身換算）',
 'c_previous_margin_code':'前走の着差コード',
 'c_previous_additional_margin_present':'前走追加着差コードの有無',
 'd_jockey_starts_365d':'騎手の過去365日出走数',
 'd_jockey_smoothed_win_365d':'騎手の過去365日平滑化勝率',
 'd_trainer_starts_365d':'調教師の過去365日出走数',
 'd_trainer_smoothed_win_365d':'調教師の過去365日平滑化勝率',
 'e_days_since_actual_start':'前回実出走からの日数',
 'e_previous_start_gap_days':'前回実出走時の出走間隔（日）',
 'e_recent3_start_gap_mean_days':'直近3回の平均出走間隔（日）',
 'e_rest_stage':'休養明けの出走段階',
 'e_rest_first_past_starts':'休養明け初戦の過去出走数',
 'e_rest_first_smoothed_finish_percentile':'休養明け初戦の平滑化成績（頭数補正）',
 'e_rest_stage_smoothed_finish_percentile':'休養段階別の平滑化成績（頭数補正）',
 'e_last_start_irregular':'前回実出走の異常フラグ',
}


def annotate(booster,metadata,train):
    rows=saved_importance(booster,metadata)
    if len(rows)!=47 or set(metadata['features'])!=set(JAPANESE) or booster.current_iteration()!=metadata['best_iteration']:
        raise safety.MLFailure('invalid_model')
    order=sorted(rows,key=lambda r:(-r['split_count'],r['feature_index']))
    split_rank={r['feature']:i for i,r in enumerate(order,1)}
    for row in rows:
        name=row['feature'];missing=train['missing'][name]
        if type(missing) is not int or not 0<=missing<=train['rows']:raise safety.MLFailure('artifact_invalid')
        row.update(japanese=JAPANESE[name],display_label=name+'（'+JAPANESE[name]+'）',
            gain_share_percent=(row.pop('gain_share') or 0.)*100,
            split_rank=split_rank[name],train_missing_count=missing,
            train_missing_percent=100*missing/train['rows'],train_all_missing=missing==train['rows'],
            extra_rest_feature=name in EXTRA,model_id=metadata['model_id'])
        row['status']='全欠損' if row['train_all_missing'] else 'Gain/Split=0' if row['gain']==0 and row['split_count']==0 else '使用あり'
    return sorted(rows,key=lambda r:r['gain_rank'])


def compare(initial,improved):
    other={r['feature']:r for r in improved}
    if set(other)!={r['feature'] for r in initial}:raise safety.MLFailure('invalid_model')
    return [dict(feature=a['feature'],japanese=a['japanese'],extra_rest_feature=a['extra_rest_feature'],
        initial_gain=a['gain'],improved_gain=other[a['feature']]['gain'],gain_difference=other[a['feature']]['gain']-a['gain'],
        initial_gain_share_percent=a['gain_share_percent'],improved_gain_share_percent=other[a['feature']]['gain_share_percent'],
        gain_share_change_percentage_points=other[a['feature']]['gain_share_percent']-a['gain_share_percent'],
        initial_gain_rank=a['gain_rank'],improved_gain_rank=other[a['feature']]['gain_rank'],
        rank_improvement=a['gain_rank']-other[a['feature']]['gain_rank'],
        initial_split=a['split_count'],improved_split=other[a['feature']]['split_count'],
        initial_split_rank=a['split_rank'],improved_split_rank=other[a['feature']]['split_rank'],
        initial_missing_percent=a['train_missing_percent'],improved_missing_percent=other[a['feature']]['train_missing_percent'],
        initial_status=a['status'],improved_status=other[a['feature']]['status']) for a in initial]


def rest_summary(rows):
    selected=[r for r in rows if r['extra_rest_feature']]
    if len(selected)!=8:raise safety.MLFailure('invalid_model')
    return {'feature_count':8,'total_gain':sum(r['gain'] for r in selected),
        'total_gain_share_percent':sum(r['gain_share_percent'] for r in selected),
        'top10_count':sum(r['gain_rank']<=10 for r in selected),'top20_count':sum(r['gain_rank']<=20 for r in selected),
        'rows':[{'feature':r['feature'],'japanese':r['japanese'],'gain_rank':r['gain_rank'],
            'gain_share_percent':r['gain_share_percent'],'gain':r['gain']} for r in selected]}
