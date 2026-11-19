param([Parameter(Mandatory=$true)][string]$RunDirectory,[switch]$CurveOnly)
$ErrorActionPreference='Stop'
try {
    Add-Type -AssemblyName System.Windows.Forms.DataVisualization
    Add-Type -AssemblyName System.Drawing
    $directory=[IO.Path]::GetFullPath($RunDirectory)
    $root=[IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'JraRacingAI\tuner-reduction'))
    if(-not $directory.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'unsafe_path'}
    $report=[IO.File]::ReadAllText((Join-Path $directory 'development_report.json'),[Text.Encoding]::UTF8)|ConvertFrom-Json
    $records=@($report.selection_records)
    if($records.Count -ne 4 -or $report.actual_new_fits -ne 16){throw 'invalid_report'}
    function New-StudyChart([string]$titleText) {
        $chart=New-Object Windows.Forms.DataVisualization.Charting.Chart
        $chart.Width=2400;$chart.Height=1450;$chart.BackColor=[Drawing.Color]::White
        $chart.AntiAliasing='All';$chart.TextAntiAliasingQuality='High'
        $area=New-Object Windows.Forms.DataVisualization.Charting.ChartArea('main')
        $chart.ChartAreas.Add($area)
        $area.Position=New-Object Windows.Forms.DataVisualization.Charting.ElementPosition(5,17,92,66)
        $area.AxisY.Title='Race Log Loss (lower is better)'
        $area.AxisY.LabelStyle.Format='F3';$area.AxisX.Interval=1
        foreach($axis in @($area.AxisX,$area.AxisY)) {
            $axis.LabelStyle.Font=New-Object Drawing.Font('Meiryo',16)
            $axis.TitleFont=New-Object Drawing.Font('Meiryo',20)
            $axis.IsLabelAutoFit=$false
            $axis.MajorGrid.LineColor=[Drawing.ColorTranslator]::FromHtml('#E6EAF0')
        }
        $title=$chart.Titles.Add($titleText)
        $title.Font=New-Object Drawing.Font('Meiryo',27,([Drawing.FontStyle]::Bold))
        $footer=$chart.Titles.Add("2024: shared temperature-fitting / strict selection window. 2025: reused development comparison.`nA10 / B20: 8 fits each. C19 / D39: saved models, zero new fits. H / Stage7 unchanged; no 2026 data.")
        $footer.Position=New-Object Windows.Forms.DataVisualization.Charting.ElementPosition(3,91,94,8)
        $footer.Font=New-Object Drawing.Font('Meiryo',15)
        return $chart
    }
    function Save-StudyChart($chart,[string]$name) {
        $file=Join-Path $directory $name
        if([IO.File]::Exists($file)){throw 'artifact_exists'}
        $bitmap=New-Object Drawing.Bitmap($chart.Width,$chart.Height)
        try {
            $chart.DrawToBitmap($bitmap,(New-Object Drawing.Rectangle(0,0,$chart.Width,$chart.Height)))
            $bitmap.SetResolution(180,180);$bitmap.Save($file,[Drawing.Imaging.ImageFormat]::Png)
        }finally{$bitmap.Dispose();$chart.Dispose()}
    }
    $barPeriods=if($CurveOnly){@()}else{@('2024','2025')}
    foreach($period in $barPeriods) {
        $chart=New-StudyChart ($period+' | four-model Race Log Loss')
        $chart.ChartAreas['main'].AxisX.Title='Model (number of features)'
        $series=New-Object Windows.Forms.DataVisualization.Charting.Series('models')
        $series.ChartType='Column';$series.IsValueShownAsLabel=$true
        $series.Font=New-Object Drawing.Font('Meiryo',22)
        $series['PointWidth']='0.50';$chart.Series.Add($series)
        $maximum=0.
        foreach($record in $records) {
            $value=if($period -eq '2024'){[double]$record.selection_metrics.race_log_loss}else{[double]$report.evaluation_metrics.PSObject.Properties[$record.name].Value.race_log_loss}
            $index=$series.Points.AddXY([string]$record.name,$value)
            $series.Points[$index].Label=('{0:F6}' -f $value)
            $series.Points[$index].Color=[Drawing.ColorTranslator]::FromHtml($(if($record.name -eq $report.overall_candidate){'#2467B3'}else{'#9CAFC3'}))
            $maximum=[Math]::Max($maximum,$value)
        }
        $chart.ChartAreas['main'].AxisY.Minimum=0
        $chart.ChartAreas['main'].AxisY.Maximum=$maximum*1.14
        $sub=$chart.Titles.Add('A10: Gain top10 | B20: Gain top20 | C19: Boruta19 | D39: all39. Blue: selected using 2024 only.')
        $sub.Font=New-Object Drawing.Font('Meiryo',15)
        Save-StudyChart $chart ('comparison_'+$period+'_race_log_loss.png')
    }
    $chart=New-StudyChart 'Feature count vs Race Log Loss | fixed four-model comparison'
    $area=$chart.ChartAreas['main'];$area.AxisX.Title='Number of original features'
    $area.AxisX.Minimum=8;$area.AxisX.Maximum=41;$area.AxisX.Interval=5;$area.AxisX.LabelStyle.Format='0'
    $legend=New-Object Windows.Forms.DataVisualization.Charting.Legend('period')
    $legend.Position=New-Object Windows.Forms.DataVisualization.Charting.ElementPosition(25,9,55,5)
    $legend.Font=New-Object Drawing.Font('Meiryo',16);$chart.Legends.Add($legend)
    $values=@()
    foreach($period in @('2024','2025')) {
        $series=New-Object Windows.Forms.DataVisualization.Charting.Series($period)
        $series.ChartType='Line';$series.BorderWidth=4;$series.MarkerStyle='Circle';$series.MarkerSize=12
        $series.IsValueShownAsLabel=$true;$series.Font=New-Object Drawing.Font('Meiryo',15)
        $series.SmartLabelStyle.Enabled=$true
        $series.Color=[Drawing.ColorTranslator]::FromHtml($(if($period -eq '2024'){'#2467B3'}else{'#AD6533'}))
        $chart.Series.Add($series)
        foreach($record in ($records|Sort-Object feature_count)) {
            $value=if($period -eq '2024'){[double]$record.selection_metrics.race_log_loss}else{[double]$report.evaluation_metrics.PSObject.Properties[$record.name].Value.race_log_loss}
            $index=$series.Points.AddXY([int]$record.feature_count,$value)
            $series.Points[$index].Label=('{0} / {1:F6}' -f $record.name,$value)
            $values+=@($value)
        }
    }
    $minimum=($values|Measure-Object -Minimum).Minimum;$maximum=($values|Measure-Object -Maximum).Maximum
    $padding=[Math]::Max(.004,($maximum-$minimum)*.2)
    $area.AxisY.Minimum=$minimum-$padding;$area.AxisY.Maximum=$maximum+$padding
    Save-StudyChart $chart 'feature_count_race_log_loss_display.png'
    if($CurveOnly){'{"status":"ok","PNG_count":1}'}else{'{"status":"ok","PNG_count":3}'}
}catch{
    '{"status":"stopped","reason":"render_failed"}'
    exit 1
}
