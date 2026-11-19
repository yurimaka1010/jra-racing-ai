param([Parameter(Mandatory=$true)][string]$BundlePath)
$ErrorActionPreference='Stop'
try {
    Add-Type -AssemblyName System.Drawing
    $full=[IO.Path]::GetFullPath($BundlePath)
    $root=[IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'JraRacingAI\model-diagnostics'))
    if(-not $full.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'unsafe_path'}
    $bundle=[IO.File]::ReadAllText($full,[Text.Encoding]::UTF8)|ConvertFrom-Json
    if($bundle.feature_count -ne 47 -or $bundle.models.Count -ne 2){throw 'invalid_input'}
    $directory=[IO.Path]::GetDirectoryName($full)
    $labelFont=New-Object Drawing.Font('Meiryo',20)
    $jpFont=New-Object Drawing.Font('Meiryo',18)
    $numberFont=New-Object Drawing.Font('Meiryo',21)
    $smallFont=New-Object Drawing.Font('Meiryo',17)
    $titleFont=New-Object Drawing.Font('Meiryo',30,([Drawing.FontStyle]::Bold))
    $blue=New-Object Drawing.SolidBrush([Drawing.ColorTranslator]::FromHtml('#2467B3'))
    $orange=New-Object Drawing.SolidBrush([Drawing.ColorTranslator]::FromHtml('#DC7C27'))
    $dark=New-Object Drawing.SolidBrush([Drawing.ColorTranslator]::FromHtml('#202B3B'))
    $muted=New-Object Drawing.SolidBrush([Drawing.ColorTranslator]::FromHtml('#556273'))
    $red=New-Object Drawing.SolidBrush([Drawing.ColorTranslator]::FromHtml('#A33A3A'))
    $grid=New-Object Drawing.Pen([Drawing.ColorTranslator]::FromHtml('#E6EAF0'),2)
    $line=New-Object Drawing.Pen([Drawing.ColorTranslator]::FromHtml('#BDC7D4'),2)
    $center=New-Object Drawing.StringFormat;$center.Alignment=[Drawing.StringAlignment]::Center
    $left=1420.;$top=420.;$plotWidth=2080.;$plotHeight=3550.;$rowHeight=$plotHeight/47.
    foreach($model in $bundle.models){
        if($model.key -notin @('next_initial','next_improved') -or $model.rows.Count -ne 47){throw 'invalid_input'}
        foreach($metric in @('gain','split')){
            $file=Join-Path $directory ($model.key+'_'+$metric+'.png')
            if([IO.File]::Exists($file)){throw 'artifact_exists'}
            $bitmap=New-Object Drawing.Bitmap(3800,4300)
            $graphics=[Drawing.Graphics]::FromImage($bitmap)
            try {
                $bitmap.SetResolution(220,220)
                $graphics.Clear([Drawing.Color]::White)
                $graphics.SmoothingMode=[Drawing.Drawing2D.SmoothingMode]::AntiAlias
                $graphics.TextRenderingHint=[Drawing.Text.TextRenderingHint]::AntiAliasGridFit
                $graphics.DrawString($model.title+' | '+$metric.ToUpperInvariant(),$titleFont,$dark,[Drawing.PointF]::new(1900,55),$center)
                $sub=('47 features | rounds={0} | T={1} | model_id={2}' -f $model.best_iteration,$model.temperature,$model.model_id)
                $graphics.DrawString($sub,$smallFont,$muted,[Drawing.PointF]::new(1900,140),$center)
                $maxValue=if($metric -eq 'gain'){[double]$bundle.chart_common_axis_max.gain}else{[double]$bundle.chart_common_axis_max.split}
                $axisMax=[Math]::Max(1.,$maxValue*1.34)
                for($tick=0;$tick -le 5;$tick++){
                    $x=$left+$plotWidth*$tick/5.
                    $graphics.DrawLine($grid,[single]$x,[single]$top,[single]$x,[single]($top+$plotHeight))
                    $graphics.DrawString(('{0:N0}' -f ($axisMax*$tick/5.)),$smallFont,$muted,[Drawing.PointF]::new($x,$top-52),$center)
                }
                $graphics.DrawLine($line,[single]$left,[single]$top,[single]$left,[single]($top+$plotHeight))
                $ordered=if($metric -eq 'gain'){@($model.rows|Sort-Object gain_rank)}else{@($model.rows|Sort-Object split_rank)}
                $rank=0
                foreach($row in $ordered){
                    $rank++
                    if($row.feature -notmatch '^[a-z][a-z0-9_]{0,100}$' -or -not $row.japanese){throw 'invalid_input'}
                    $value=[double]$(if($metric -eq 'gain'){$row.gain}else{$row.split_count})
                    if([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0){throw 'invalid_input'}
                    $y=$top+($rank-.5)*$rowHeight
                    $barBrush=if($row.extra_rest_feature){$orange}else{$blue}
                    $tag=if($row.train_all_missing){' [ALL NA]'}elseif($value -eq 0){' [ZERO]'}else{''}
                    $english=('{0:00}  {1}' -f $rank,$row.feature)
                    $japanese='    ('+$row.japanese+')'+$tag
                    if($graphics.MeasureString($english,$labelFont).Width -gt ($left-110) -or $graphics.MeasureString($japanese,$jpFont).Width -gt ($left-110)){throw 'label_overflow'}
                    $graphics.DrawString($english,$labelFont,$barBrush,60,[single]($y-34))
                    $graphics.DrawString($japanese,$jpFont,$muted,90,[single]($y+2))
                    $width=$plotWidth*$value/$axisMax
                    if($value -gt 0){$graphics.FillRectangle($barBrush,[single]$left,[single]($y-17),[single]$width,34)}
                    $number=if($metric -eq 'gain'){('{0:N4}  |  {1:F2}%' -f $value,$row.gain_share_percent)}else{('{0:N0}' -f $value)}
                    $ink=if($value -eq 0){$red}else{$dark}
                    $numberX=$left+$width+14
                    if($numberX+$graphics.MeasureString($number,$numberFont).Width -gt 3770){throw 'number_overflow'}
                    $graphics.DrawString($number,$numberFont,$ink,[single]$numberX,[single]($y-20))
                }
                if($rank -ne 47){throw 'invalid_input'}
                $note=if($metric -eq 'gain'){$bundle.gain_note}else{$bundle.split_note}
                $graphics.DrawString($bundle.legend,$smallFont,$muted,70,4050)
                $graphics.DrawString($note,$smallFont,$muted,70,4110)
                $graphics.DrawString($bundle.interpretation_note,$smallFont,$muted,70,4170)
                $bitmap.Save($file,[Drawing.Imaging.ImageFormat]::Png)
            } finally {$graphics.Dispose();$bitmap.Dispose()}
        }
    }
    foreach($item in @($labelFont,$jpFont,$numberFont,$smallFont,$titleFont,$blue,$orange,$dark,$muted,$red,$grid,$line,$center)){$item.Dispose()}
    '{"status":"ok","PNG_count":4,"labels_per_chart":47,"untruncated_labels":true,"font":"Meiryo","width":3800,"height":4300}'
}catch{
    '{"status":"stopped","reason":"render_failed"}'
    exit 1
}