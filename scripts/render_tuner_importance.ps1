param([Parameter(Mandatory=$true)][string]$BundlePath)
$ErrorActionPreference = 'Stop'
try {
    Add-Type -AssemblyName System.Windows.Forms.DataVisualization
    Add-Type -AssemblyName System.Drawing
    $full = [IO.Path]::GetFullPath($BundlePath)
    $root = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'JraRacingAI\model-diagnostics'))
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'unsafe_path' }
    $bundle = [IO.File]::ReadAllText($full, [Text.Encoding]::UTF8) | ConvertFrom-Json
    if ($bundle.feature_count -ne 39 -or $bundle.models.Count -ne 2) { throw 'invalid_input' }
    $directory = [IO.Path]::GetDirectoryName($full)
    $font = New-Object Drawing.Font('Meiryo', 20)
    $small = New-Object Drawing.Font('Meiryo', 16)
    $heading = New-Object Drawing.Font('Meiryo', 28, ([Drawing.FontStyle]::Bold))
    $numbers = New-Object Drawing.Font('Meiryo', 18)
    foreach ($model in $bundle.models) {
        if ($model.key -notin @('tuner','binary_control') -or $model.rows.Count -ne 39) { throw 'invalid_input' }
        foreach ($metric in @('gain','split')) {
            $chart = New-Object Windows.Forms.DataVisualization.Charting.Chart
            try {
                $chart.Width = 3400; $chart.Height = 2800
                $chart.BackColor = [Drawing.Color]::White
                $chart.AntiAliasing = 'All'; $chart.TextAntiAliasingQuality = 'High'
                $area = New-Object Windows.Forms.DataVisualization.Charting.ChartArea('main')
                $chart.ChartAreas.Add($area)
                $area.Position = New-Object Windows.Forms.DataVisualization.Charting.ElementPosition(0,14,99,77)
                $area.InnerPlotPosition = New-Object Windows.Forms.DataVisualization.Charting.ElementPosition(30,1,68,96)
                $area.AxisX.Minimum = 0.5; $area.AxisX.Maximum = 39.5
                $area.AxisX.Interval = 1; $area.AxisX.IsReversed = $true
                $area.AxisX.Enabled = 'True'; $area.AxisX.LabelStyle.Enabled = $true
                $area.AxisX.IsLabelAutoFit = $false; $area.AxisX.LabelStyle.Font = $font
                $area.AxisX.MajorGrid.Enabled = $false; $area.AxisX.MajorTickMark.Enabled = $false
                $area.AxisX.LabelStyle.TruncatedLabels = $false
                $area.AxisX.LineColor = [Drawing.ColorTranslator]::FromHtml('#D4DAE5')
                $area.AxisY.Minimum = 0; $area.AxisY.IsLabelAutoFit = $false
                $area.AxisY.LabelStyle.Font = $small; $area.AxisY.LabelStyle.Format = '#,##0'
                $area.AxisY.MajorGrid.LineColor = [Drawing.ColorTranslator]::FromHtml('#E6EAF0')
                $area.AxisY.LineColor = [Drawing.ColorTranslator]::FromHtml('#D4DAE5')
                $series = New-Object Windows.Forms.DataVisualization.Charting.Series('importance')
                $series.ChartType = 'Bar'; $series.IsValueShownAsLabel = $true
                $series.Font = $numbers; $series['BarLabelStyle'] = 'Outside'
                $series['PixelPointWidth'] = '34'; $series.SmartLabelStyle.Enabled = $false
                $series.Color = [Drawing.ColorTranslator]::FromHtml($(if ($model.key -eq 'tuner') {'#2467B3'} else {'#AD6533'}))
                $chart.Series.Add($series)
                if ($metric -eq 'gain') { $ordered = @($model.rows | Sort-Object gain_rank) }
                else { $ordered = @($model.rows | Sort-Object split_rank) }
                $maximum = 0.0; $rank = 0
                foreach ($row in $ordered) {
                    $rank++
                    if ($row.feature -notmatch '^[a-z][a-z0-9_]{0,100}$') { throw 'invalid_input' }
                    $value = [double]$(if ($metric -eq 'gain') { $row.gain } else { $row.split_count })
                    if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0) { throw 'invalid_input' }
                    $i = $series.Points.AddXY($rank, $value); $point = $series.Points[$i]
                    $tag = ''
                    if ($row.train_all_missing) { $tag = ' [ALL NA]' }
                    elseif ($value -eq 0) { $tag = ' [ZERO]' }
                    $point.AxisLabel = ('{0:00}  {1}{2}' -f $rank, $row.feature, $tag)
                    [void]$area.AxisX.CustomLabels.Add(($rank - 0.5), ($rank + 0.5), $point.AxisLabel)
                    if ($metric -eq 'gain') { $point.Label = ('{0:N4}  |  {1:F2}%' -f $value, $row.gain_share_percent) }
                    else { $point.Label = ('{0:N0}' -f $value) }
                    if ($value -eq 0) { $point.LabelForeColor = [Drawing.ColorTranslator]::FromHtml('#A33A3A') }
                    $maximum = [Math]::Max($maximum, $value)
                }
                $area.AxisY.Maximum = [Math]::Max(1.0, $maximum * 1.30)
                $title = $chart.Titles.Add($model.title + ' | ' + $metric.ToUpperInvariant() + ' importance')
                $title.Font = $heading; $title.Docking = 'Top'
                $sub = $chart.Titles.Add(('39 features | best_iteration={0} | model_id={1}' -f $model.best_iteration, $model.model_id))
                $sub.Font = $small; $sub.Docking = 'Top'
                $note = 'ALL NA = all training values missing; ZERO = zero importance (not all missing).'
                if ($metric -eq 'gain') { $note += "`nLabels: raw Gain (4 decimals) | share of total Gain (%). CSV retains full precision." }
                else { $note += "`nLabels: actual split counts. All 39 features, including zeros, are retained." }
                $note += "`nSaved-tree training importance; not causal contribution or out-of-sample improvement. No refitting."
                $footer = $chart.Titles.Add($note); $footer.Font = $small; $footer.Docking = 'Bottom'
                $file = Join-Path $directory ($model.key + '_' + $metric + '.png')
                if ([IO.File]::Exists($file)) { throw 'artifact_exists' }
                $image = New-Object Drawing.Bitmap($chart.Width, $chart.Height)
                try {
                    $chart.DrawToBitmap($image, (New-Object Drawing.Rectangle(0,0,$chart.Width,$chart.Height)))
                    $image.SetResolution(180,180)
                    $image.Save($file, [Drawing.Imaging.ImageFormat]::Png)
                } finally { $image.Dispose() }
            } finally { $chart.Dispose() }
        }
    }
    $font.Dispose(); $small.Dispose(); $heading.Dispose(); $numbers.Dispose()
    '{"status":"ok","PNG_count":4}'
} catch {
    '{"status":"stopped","reason":"render_failed"}'
    exit 1
}
