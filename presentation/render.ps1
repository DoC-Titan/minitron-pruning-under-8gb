$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
$Slides=Get-Content (Join-Path $PSScriptRoot 'slides.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$HadPowerPoint=@(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$Preview=Join-Path $PSScriptRoot 'previews'
New-Item -ItemType Directory -Force -Path $Preview | Out-Null
$App=$null; $Deck=$null
$Warnings=New-Object System.Collections.Generic.List[object]
function Color($hex) {
 $h=$hex.TrimStart('#')
 return [Convert]::ToInt32($h.Substring(0,2),16)+256*[Convert]::ToInt32($h.Substring(2,2),16)+65536*[Convert]::ToInt32($h.Substring(4,2),16)
}
function Text($s,$x,$y,$w,$h,$str,$size,$color='#1B3044',$bold=$false) {
 $t=$s.Shapes.AddTextbox(1,$x,$y,$w,$h)
 $t.TextFrame.MarginLeft=0; $t.TextFrame.MarginRight=0; $t.TextFrame.MarginTop=0; $t.TextFrame.MarginBottom=0
 $t.TextFrame.WordWrap=-1; $t.TextFrame.AutoSize=0
 $t.TextFrame.TextRange.Text=$str
 $t.TextFrame.TextRange.Font.Name='Aptos'
 $t.TextFrame.TextRange.Font.Size=$size
 $t.TextFrame.TextRange.Font.Color.RGB=(Color $color)
 $t.TextFrame.TextRange.Font.Bold=$(if($bold){-1}else{0})
 $t.TextFrame.TextRange.ParagraphFormat.SpaceAfter=10
 $t.TextFrame2.AutoSize=0
 $t.Height=$h; $t.Width=$w
 return $t
}
try {
 $App=New-Object -ComObject PowerPoint.Application
 $Deck=$App.Presentations.Add(0)
 $Deck.PageSetup.SlideWidth=960; $Deck.PageSetup.SlideHeight=540
 $i=0
 foreach($spec in $Slides) {
  $i++; $s=$Deck.Slides.Add($i,12)
  $s.FollowMasterBackground=0; $s.Background.Fill.Solid()
  $dark=$i -eq 1
  $s.Background.Fill.ForeColor.RGB=(Color $(if($dark){'#102B3B'}else{'#F6F8FA'}))
  $ink=$(if($dark){'#FFFFFF'}else{'#173347'})
  $bar=$s.Shapes.AddShape(1,36,35,42,5); $bar.Fill.ForeColor.RGB=(Color '#00A2A0'); $bar.Line.Visible=0
  [void](Text $s 36 52 890 25 $spec.kicker 11 $(if($dark){'#71D4D0'}else{'#007F83'}) $true)
  [void](Text $s 36 88 890 80 $spec.title $(if($dark){37}else{29}) $ink $true)
  $body=($spec.body | ForEach-Object { [string]$_ }) -join "`r`n"
  if($spec.figure) {
   $size=$(if($spec.body.Count -gt 4){14}else{16})
   [void](Text $s 36 182 330 298 $body $size $ink)
   $image=Join-Path $Root ('figures\'+$spec.figure)
   [void]$s.Shapes.AddPicture($image,0,-1,381,182,545,267)
   [void](Text $s 387 456 530 32 'Measured local data. Raw timings and per-seed results are retained.' 10 '#647581')
  } else {
   [void](Text $s 48 190 855 292 $body $(if($i -eq 1){24}else{21}) $ink)
  }
  [void](Text $s 36 511 820 16 'MINITRON-INSPIRED  /  BUILT WITH QWEN  /  LOCAL MEASUREMENTS, NOT NVIDIA RESULTS' 8 $(if($dark){'#A8C4CC'}else{'#647581'}))
  [void](Text $s 889 506 38 22 ('{0:00}' -f $i) 12 $(if($dark){'#A8C4CC'}else{'#647581'}))
  $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$spec.notes
  foreach($shape in @($s.Shapes)) {
   if($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
    $needed=$shape.TextFrame2.TextRange.BoundHeight
    if($needed -gt ($shape.Height+4)) { $Warnings.Add(@{slide=$i; needed=$needed; available=$shape.Height; text=$shape.TextFrame.TextRange.Text}) }
   }
   if($shape.Left -lt 0 -or $shape.Top -lt 0 -or $shape.Left+$shape.Width -gt 962 -or $shape.Top+$shape.Height -gt 542) {
    $Warnings.Add(@{slide=$i;type='out-of-bounds';name=$shape.Name})
   }
  }
  $s.Export((Join-Path $Preview ('slide-{0:00}.png' -f $i)),'PNG',1280,720)
 }
 $Deck.SaveAs((Join-Path $PSScriptRoot 'Pruning_Under_8GiB_Level4.pptx'),24)
 $Deck.SaveCopyAs((Join-Path $PSScriptRoot 'Pruning_Under_8GiB_Level4.pdf'),32)
 $Warnings | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $PSScriptRoot 'LAYOUT_WARNINGS.json') -Encoding UTF8
 @{slide_count=$Deck.Slides.Count;layout_warnings=$Warnings.Count;status=$(if($Warnings.Count -eq 0){'PASS'}else{'REVIEW_REQUIRED'})} | ConvertTo-Json | Set-Content (Join-Path $PSScriptRoot 'VALIDATION.json') -Encoding UTF8
 if($Deck.Slides.Count -ne 12 -or $Warnings.Count -gt 0) { throw 'Presentation needs layout review; not claiming a validated deck.' }
} finally {
 if($Deck) { $Deck.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($Deck) }
 if($App) { if(-not $HadPowerPoint) { $App.Quit() }; [void][Runtime.InteropServices.Marshal]::ReleaseComObject($App) }
 [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
