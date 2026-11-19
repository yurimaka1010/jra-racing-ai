param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
try {
    Add-Type -AssemblyName System.Windows.Forms.DataVisualization
    Add-Type -AssemblyName System.Drawing
    $directory=[IO.Path]::GetFullPath($RunDirectory)
    $root=[IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'JraRacingAI\tuner-reduction'))
    if (-not $directory.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'unsafe_path'}
    $report=[IO.File]::ReadAllText((Join-Path $directory 'development_report.json'),[Text.Encoding]::UTF8)|ConvertFrom-Json
    foreach($scope in @('selection_2024','frozen_2025')) {
        $chart=New-Object Windows.Forms.DataVisualization.Charting.Chart
        try {
            $chart.Width=2400;$chart.Height=1450;$chart.BackColor=[Drawing.Color]::White
            $chart.AntiAliasing='All';$chart.TextAntiAliasingQuality='High'
            $area=New-Object Windows.Forms.DataVisualization.Charting.ChartArea('main')
            $chart.ChartAreas.Add($area)
            $area.Position=New-Object Windows.Forms.DataVisualization.Charting.ElementPosition(4,17,92,67)
            $area.AxisX.Title='Number of original features';$area.AxisY.Title='Race Log Loss (lower is better)'
            $area.AxisX.Interval=1;$area.AxisX.IsReversed=$true
            $area.AxisY.LabelStyle.Format='F3'
            foreach($axis in @($area.AxisX,$area.AxisY)) {
                $axis.LabelStyle.Font=New-Object Drawing.Font('Meiryo',15)
                $axis.TitleFont=New-Object Drawing.Font('Meiryo',18)
                $axis.IsLabelAutoFit=$false
                $axis.MajorGrid.LineColor=[Drawing.ColorTranslator]::FromHtml('#E6EAF0')
            }
            $all=@($report.selection_records)
            if($report.Boruta_result) {$all+=@($report.Boruta_result)}
            $values=@();$counts=@()
            foreach($family in @('backward','Boruta')) {
                $series=New-Object Windows.Forms.DataVisualization.Charting.Series($family)
                $series.ChartType='Line';$series.BorderWidth=4;$series.MarkerStyle='Circle';$series.MarkerSize=12
                $series.IsValueShownAsLabel=$true;$series.Font=New-Object Drawing.Font('Meiryo',14)
                $series.SmartLabelStyle.Enabled=$true
                $series.Color=[Drawing.ColorTranslator]::FromHtml($(if($family -eq 'backward'){'#2467B3'}else{'#AD6533'}))
                $chart.Series.Add($series)
                foreach($record in $all) {
                    if(($family -eq 'Boruta') -ne ($record.name -eq 'Boruta_C')) {continue}
                    $value=$null
                    if($scope -eq 'selection_2024') {$value=[double]$record.selection_metrics.race_log_loss}
                    else {
                        $property=$report.evaluation_metrics.PSObject.Properties[$record.name]
                        if($null -ne $property) {$value=[double]$property.Value.race_log_loss}
                    }
                    if($null -eq $value) {continue}
                    $index=$series.Points.AddXY([int]$record.feature_count,$value)
                    $series.Points[$index].Label=('{0} / {1:F6}' -f $record.feature_count,$value)
                    $values+=@($value);$counts+=@([int]$record.feature_count)
                }
            }
            $minimum=($values|Measure-Object -Minimum).Minimum;$maximum=($values|Measure-Object -Maximum).Maximum
            $padding=[Math]::Max(.004,($maximum-$minimum)*.30)
            $area.AxisY.Minimum=$minimum-$padding;$area.AxisY.Maximum=$maximum+$padding
            $area.AxisX.Minimum=($counts|Measure-Object -Minimum).Minimum-1
            $area.AxisX.Maximum=40;$area.AxisX.LabelStyle.Format='0'
            $area.AxisX.Crossing=40;$area.AxisY.Crossing=$area.AxisY.Minimum
            $title=$chart.Titles.Add($scope+' | feature count vs Race Log Loss')
            $title.Font=New-Object Drawing.Font('Meiryo',25,([Drawing.FontStyle]::Bold))
            $sub=$chart.Titles.Add('Labels: number of features / measured loss. Blue: A39 and backward; brown: Boruta.')
            $sub.Font=New-Object Drawing.Font('Meiryo',15)
            $note=if($scope -eq 'selection_2024') {'2024 development selection window; also used for temperature fitting. Not an untouched test.'}else{'2025 reused development comparison; only candidates frozen before evaluation are shown.'}
            $note+="`nSame 8-fit Tuner budget per new configuration. No H changes; no sealed 2026 or Stage 7 outcomes."
            $footer=$chart.Titles.Add($note);$footer.Docking='Bottom';$footer.Font=New-Object Drawing.Font('Meiryo',15)
            $file=Join-Path $directory ($scope+'_race_log_loss.png')
            if([IO.File]::Exists($file)){throw 'artifact_exists'}
            $image=New-Object Drawing.Bitmap($chart.Width,$chart.Height)
            try {
                $chart.DrawToBitmap($image,(New-Object Drawing.Rectangle(0,0,$chart.Width,$chart.Height)))
                $image.SetResolution(180,180);$image.Save($file,[Drawing.Imaging.ImageFormat]::Png)
            }finally{$image.Dispose()}
        }finally{$chart.Dispose()}
    }
    '{"status":"ok","PNG_count":2}'
}catch{
    '{"status":"stopped","reason":"render_failed"}'
    exit 1
}
